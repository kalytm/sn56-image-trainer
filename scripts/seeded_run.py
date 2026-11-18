"""Run AI Toolkit's run.py with every RNG seeded from G56_SEED (D12), so seeded members differ in LoRA init, noise
and timestep draws. A fresh torch process otherwise starts from the same default seed, which would make members
identical. Usage (cwd = the AI Toolkit root): python3 seeded_run.py <config.yaml>"""
import os
import random
import runpy
import sys


def main():
    seed = int(os.environ["G56_SEED"])
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    sys.argv = ["run.py", *sys.argv[1:]]
    sys.path.insert(0, os.getcwd())
    runpy.run_path("run.py", run_name="__main__")


if __name__ == "__main__":
    main()
