TrackDiff.cu.exe \
    --gpu-nb-blocks=1024 \
    --gpu-core=1 \
    --edge-layer-connector-min-hits=5 \
    --max-hits-per-track=30 \
    --edge-layer-connector-weights-cut=0.01 \
    --input-dir=/scratch_ssd/collard/dump_pyg \ #/tmp/1dump_pyg \ # /scratch_ssd/collard/1dump_pyg \ # \ #/sps/atlas/s/scaillou/Christophe \ #1event.walkthrough \
    --input-filename-pattern=csv \
    --output-dir=/scratch_ssd/collard/OUTPUT_DIR_GEN \ #/tmp/OUTPUT_DIR_GEN \ # /scratch_ssd/collard/OUTPUT_DIR_GEN \
