# AMD scripts

Save the first self-attention Q/K/V from video generation:

```bash
sbatch amd_scripts/save_first_qkv.sh
```

Run the BM41 vs Monarch vs dense QKV attention experiment:

```bash
sbatch amd_scripts/run_qkv_attention_experiment.sh
```

The scripts follow the same AMDHPC style as the existing self-forcing script:
Slurm headers, conda discovery, `conda activate monarch_rt`, then one explicit
`python ...` command.
