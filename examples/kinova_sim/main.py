
"""CLI to test Kinova Gazebo simulation with PI‑0 environment."""
import numpy as np
from examples.kinova_sim import env as ks_env

def main():
    env = ks_env.SimEnv()
    ts = env.reset()
    done = False
    while not done:
        action = np.zeros((8,), dtype=np.float32)
        ts = env.step(action)
        done = ts.last()

if __name__ == '__main__':
    main()
