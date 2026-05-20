# AMD scripts

Submit BM41 inference:

```bash
sbatch amd_scripts/run_bm41_inference.sh
```

Submit BM41 QKV comparison:

```bash
sbatch amd_scripts/run_bm41_compare.sh
```

The scripts follow the same AMDHPC style as the existing self-forcing script:
Slurm headers, conda discovery, `conda activate monarch_rt`, then one explicit
`python ...` command.
