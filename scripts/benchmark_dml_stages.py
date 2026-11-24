"""Per-stage latency and track count of DoubleMetricLearning versions 1 and 2.

Times embedding, build-edges, filtering, gnn and graph segmentation by wrapping each
stage's function in the version's inference.py with torch.cuda.synchronize() on both
sides, so the model code is unchanged. "gap" = total - sum of stages: data movement and
bookkeeping between stages (e.g. v1's graph construction and per-track output loop).
"outputs" is the number of distinct track outputs over the runs; 1 means reproducible.
With --deterministic, every row is followed by the same configuration with the GNN
message sums in a fixed order (the `deterministic` model parameter).

Each (version, backend) runs in its own process, because both versions use the same module
names. Run on a GPU compute node, not a login node, e.g. on Perlmutter:

    salloc -N 1 -C gpu -q interactive -t 30 -A m3443 --gpus 1
    podman-hpc run --rm --gpu -v $PWD:$PWD -v $PSCRATCH:$PSCRATCH -w $PWD \
        <image with frnn, frnn_cuda and graph_segment> \
        python scripts/benchmark_dml_stages.py -i all_input_node_features.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "DoubleMetricLearning"
STAGES = ["embedding", "build-edges", "filtering", "gnn", "graph segmentation"]
BACKENDS = ["cc_walk", "cc", "cc_bestpred_walk", "cc_junction_removal"]


def worker(
    version: str,
    backend: str | None,
    deterministic: bool,
    input_path: str,
    warmup: int,
    trials: int,
):
    import os

    import torch

    model_dir = MODEL_DIR / version
    sys.path.insert(0, str(model_dir))
    os.chdir(model_dir)
    import inference as m

    times = {}

    def timed(stage, fn):
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            torch.cuda.synchronize()
            times[stage] = times.get(stage, 0.0) + (time.perf_counter() - start) * 1e3
            return result

        return wrapper

    # run_torch_model is called twice per event: the embedding, then the gnn.
    run_torch_model = m.run_torch_model

    def run_torch_model_timed(*args, **kwargs):
        stage = "gnn" if "embedding" in times else "embedding"
        return timed(stage, run_torch_model)(*args, **kwargs)

    m.run_torch_model = run_torch_model_timed
    m.build_edges = timed("build-edges", m.build_edges)
    m.run_gnn_filter = timed("filtering", m.run_gnn_filter)
    if version == "1":  # cc and walk are two calls
        m.walkutils.get_simple_path = timed("graph segmentation", m.walkutils.get_simple_path)
        m.walkutils.walk_through = timed("graph segmentation", m.walkutils.walk_through)
    else:
        m.segment = timed("graph segmentation", m.segment)

    kwargs = dict(model_path=".", device="cuda", debug=False, precision="high",
                  auto_cast=False, compiling=False, save_data_for_debug=False,
                  deterministic=deterministic)  # fmt: skip
    if version == "1":
        kwargs["use_dpw"] = False
    else:
        kwargs["segment_backend"] = backend
    inference = m.create_metric_learning_end2end_rel24(**kwargs)

    node_features = torch.load(input_path)
    for _ in range(warmup):
        inference(node_features)
    runs = []
    for _ in range(trials):
        times.clear()
        torch.cuda.synchronize()
        start = time.perf_counter()
        track_candidates = inference(node_features)
        torch.cuda.synchronize()
        times["total"] = (time.perf_counter() - start) * 1e3
        times["tracks"] = int(np.sum(track_candidates == -1))
        times["hash"] = hashlib.sha1(np.ascontiguousarray(track_candidates).tobytes()).hexdigest()
        runs.append(dict(times))
    print("RESULT " + json.dumps(runs))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-i", "--input", required=True, help="node features .pt file")
    parser.add_argument("-b", "--backends", nargs="+", default=BACKENDS,
                        help="graph_segment backends to run with version 2")  # fmt: skip
    parser.add_argument("-w", "--warmup", type=int, default=2)
    parser.add_argument("-n", "--trials", type=int, default=10)
    parser.add_argument("-D", "--deterministic", action="store_true",
                        help="also run each configuration with deterministic GNN sums")  # fmt: skip
    parser.add_argument(
        "--worker", nargs=3, metavar=("VERSION", "BACKEND", "DET"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    input_path = str(Path(args.input).resolve())

    if args.worker:
        version, backend, det = args.worker
        backend = None if backend == "-" else backend
        worker(version, backend, det == "1", input_path, args.warmup, args.trials)
        return

    print("| version | " + " | ".join(STAGES) + " | sum | gap | total | tracks | outputs |")
    print("|---" * (len(STAGES) + 6) + "|")
    configs = [("1", "-")] + [("2", b) for b in args.backends]
    dets = ["0", "1"] if args.deterministic else ["0"]
    for (version, backend), det in [(c, d) for c in configs for d in dets]:
        cmd = [sys.executable, __file__, "-i", input_path, "-w", str(args.warmup),
               "-n", str(args.trials), "--worker", version, backend, det]  # fmt: skip
        out = subprocess.run(cmd, capture_output=True, text=True)
        lines = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
        if out.returncode or not lines:
            sys.exit(f"v{version} {backend} failed:\n{out.stderr[-3000:]}")
        runs = json.loads(lines[-1][len("RESULT ") :])
        med = {k: float(np.median([r.get(k, 0.0) for r in runs])) for k in [*STAGES, "total"]}
        stage_sum = sum(med[k] for k in STAGES)
        tracks = sorted({r["tracks"] for r in runs})
        label = ("v1" if version == "1" else f"v2 `{backend}`") + (" det" if det == "1" else "")
        print(f"| {label} | " + " | ".join(f"{med[k]:.1f}" for k in STAGES)
              + f" | {stage_sum:.1f} | {med['total'] - stage_sum:.1f}"
              + f" | {med['total']:.1f} ± {np.std([r['total'] for r in runs]):.1f}"
              + f" | {', '.join(map(str, tracks))} | {len({r['hash'] for r in runs})} |",
              flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
