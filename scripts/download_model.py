#!/usr/bin/env python
from pathlib import Path

import requests
import tqdm


def download():
    """Download models from NERSC portal."""
    print("Downloading models from NERSC portal...")

    # use tqdm to show the download progress.
    # the file is located at https://portal.nersc.gov/cfs/m3443/xju/trition_server

    file_base_url = "https://portal.nersc.gov/cfs/m3443/xju/trition_server/"
    file_dirs = ["GNN4Pixel", "MetricLearning"]
    version = "1"

    # download the model files
    model_names = ["embedding.pt", "filter.pt", "gnn.pt"]
    this_file_path = Path(__file__).resolve()
    for dir_name in file_dirs:
        for model_name in model_names:
            url = file_base_url + f"{dir_name}/{version}/{model_name}"
            response = requests.get(str(url), stream=True, timeout=200)
            out_name = this_file_path.parent.parent / "models" / dir_name / version / model_name
            if out_name.exists():
                print(f"{out_name} already exists. Skip downloading.")
                continue

            for data in tqdm.tqdm(response.iter_content()):
                out_name.write_bytes(data)

    print("Download complete.")


if __name__ == "__main__":
    download()
