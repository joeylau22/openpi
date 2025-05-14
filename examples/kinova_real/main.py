
"""CLI launcher for Kinova PI‑0 client."""
import argparse, numpy as np
from examples.kinova_real import kinova_env as kv_env

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', default='reach')
    args = ap.parse_args()
    env = kv_env.KinovaRealEnvironment()
    timestep = env.reset()
    done = False
    while not done:
        action = np.zeros((8,), dtype=np.float32)
        timestep = env.step(action)
        done = timestep.last()
if __name__ == '__main__':
    main()
