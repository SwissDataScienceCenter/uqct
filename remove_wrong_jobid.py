import os
from glob import glob

import pandas as pd

dryrun = False
files = sorted(glob("./results/runs/*.parquet"))
valid_jobids = [54717753, 54757527]

for file in files:
    df = pd.read_parquet(file)
    jobid = int(df["slurm_job_id"][0])  # type: ignore
    if jobid not in valid_jobids:
        if dryrun:
            print(f"Removing: {file}")
        else:
            os.remove(file)
