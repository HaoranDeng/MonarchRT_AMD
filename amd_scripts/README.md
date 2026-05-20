# AMD / BM41 helper scripts

These scripts are thin wrappers for running the AMD-friendly BM41 attention path.

## BM41 QKV comparison

```bash
bash amd_scripts/run_bm41_compare.sh
```

Useful environment overrides:

```bash
NPZ=assets/first_qkv/first_attn_qkv_dense_layer0_ts999.npz \
DEVICE=cuda \
BLOCK_SIZE=32 \
QUICK_TOKENS=4096 \
bash amd_scripts/run_bm41_compare.sh
```

## BM41 inference

```bash
CHECKPOINT_PATH=checkpoints/model.pt \
DATA_PATH=prompts/MovieGenVideoBench.txt \
OUTPUT_FOLDER=videos_bm41 \
bash amd_scripts/run_bm41_inference.sh
```

The script uses `configs/wan_bm41_fewstep_dmd.yaml`, where:

```yaml
bm41_args:
  enable: true
  block_size: 32
```

Override `CONFIG_PATH`, `CHECKPOINT_PATH`, `DATA_PATH`, `OUTPUT_FOLDER`, `MAX_PROMPTS`, and `NUM_SAMPLES` as needed.
