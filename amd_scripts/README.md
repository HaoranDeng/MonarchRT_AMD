# AMD / BM41 helper scripts

These scripts follow the AMDHPC Slurm style used for MonarchRT runs:

- `#SBATCH -p mi3001x`
- `conda activate monarch_rt`
- logs under `out/job.%j.out` and `out/job.%j.err`

## BM41 QKV comparison

```bash
sbatch amd_scripts/run_bm41_compare.sh
```

Useful environment overrides:

```bash
sbatch --export=ALL,NPZ=assets/first_qkv/first_attn_qkv_dense_layer0_ts999.npz,DEVICE=cuda,BLOCK_SIZE=32,QUICK_TOKENS=4096 amd_scripts/run_bm41_compare.sh
```

## BM41 inference

```bash
sbatch amd_scripts/run_bm41_inference.sh
```

Useful overrides:

```bash
sbatch --export=ALL,CHECKPOINT_PATH=checkpoints/self_forcing_dmd.pt,DATA_PATH=prompts/MovieGenVideoBench_extended.txt,OUTPUT_FOLDER=videos/bm41 amd_scripts/run_bm41_inference.sh
```

The inference script uses `configs/wan_bm41_fewstep_dmd.yaml`, where:

```yaml
bm41_args:
  enable: true
  block_size: 32
```

Override `CONDA_ENV`, `CONFIG_PATH`, `CHECKPOINT_PATH`, `DATA_PATH`, `OUTPUT_FOLDER`, `MAX_PROMPTS`, `NUM_SAMPLES`, `SEED`, and `SAVE_WITH_INDEX` as needed.
