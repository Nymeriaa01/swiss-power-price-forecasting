"""Download all raw data into data/raw (cached, re-run with --force to refresh)."""
import sys

from _setup import *  # noqa: F401,F403  (adds src/ to the path)
from swisspower import data

if __name__ == "__main__":
    d = data.load_all(force="--force" in sys.argv)
    for name, obj in d.items():
        print(f"{name:8s} {obj.shape}  {obj.index.min()} -> {obj.index.max()}  missing={int(obj.isna().sum().sum())}")
