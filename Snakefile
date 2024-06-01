import os

os.environ["JULIA_PROJECT"] = "."
os.environ["JULIA_NUM_THREADS"] = "auto"

envvars:
    "JULIA_PROJECT",
    "JULIA_NUM_THREADS",