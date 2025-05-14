
"""KinovaPolicy – duplicate of PI‑0 aloha_policy but with 7+1 dims."""
import numpy as np
from typing import Dict

STATE_DIM = 8
ACTION_DIM = 8
EXPECTED_CAMERAS = ['color']

class KinovaPolicy:
    def __init__(self, model_path: str):
        # TODO: load actual model
        self.model_path = model_path

    def __call__(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        return np.zeros((ACTION_DIM,), dtype=np.float32)
