# AMD scripts

Run BM41 inference:

```bash
sbatch amd_scripts/bm41.sh
```

Save the first self-attention Q/K/V from video generation:

```bash
sbatch amd_scripts/save_first_qkv.sh
```

Run the BM41 vs Monarch vs dense QKV attention experiment:

```bash
sbatch amd_scripts/run_qkv_attention_experiment.sh
```

Run the four BM41 initial-q video experiments on Mi3001:

```bash
sbatch amd_scripts/run_q_init_experiments.sh
```

This writes videos to `videos/mean`, `videos/random`, `videos/1st`, and
`videos/ith`. `ith` matches the MRT-1 identity initial-q choice.

Run the contiguous-query block experiment on Mi3001:

```bash
sbatch amd_scripts/run_contiguous_q_experiment.sh
```

This writes videos to `videos/contiguous_ith`.

BM41 now uses the same `f_tied`, `h_reduce`, and `w_reduce` block layout as
one-iteration MonarchRT; these are configured in `bm41_args`.

The scripts follow the same AMDHPC style as the existing self-forcing script:
Slurm headers, conda discovery, `conda activate monarch_rt`, then one explicit
`python ...` command.
