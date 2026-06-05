# AMD scripts

Run MonarchRT inference:

```bash
sbatch amd_scripts/monarch_rt.sh
```

Save the first self-attention Q/K/V from video generation:

```bash
sbatch amd_scripts/save_first_qkv.sh
```

Run the four Monarch initial-q video experiments on Mi3001:

```bash
sbatch amd_scripts/run_monarch_q_init_experiments.sh
```

This writes videos to `videos/mean`, `videos/random`, `videos/1st`, and
`videos/ith`. `ith` matches the original MRT-1 identity initial-q choice.

The scripts follow the same AMDHPC style as the existing self-forcing script:
Slurm headers, conda discovery, `conda activate monarch_rt`, then one explicit
`python ...` command.
