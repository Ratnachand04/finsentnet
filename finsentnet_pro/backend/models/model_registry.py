"""
FINSENT NET PRO - Model Registry
In-memory model profile catalog for runtime configuration and UI introspection.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class ModelProfile:
    profile_id: str
    display_name: str
    model_class: str
    version: str
    price_window: int
    price_features: int
    text_tokens: int
    output_heads: List[str]
    checkpoint_pattern: str
    notes: str


class ModelRegistry:
    """Keeps track of available model runtime profiles."""

    def __init__(self):
        self._profiles: Dict[str, ModelProfile] = {
            "core-v1": ModelProfile(
                profile_id="core-v1",
                display_name="FinSent Core v1",
                model_class="FinSentNetCore",
                version="1.0.0",
                price_window=30,
                price_features=20,
                text_tokens=50,
                output_heads=["direction_classification", "return_regression"],
                checkpoint_pattern="checkpoints/{ticker}_best_model.pth",
                notes="Stable default profile for live prediction and training routes.",
            ),
            "core-lite": ModelProfile(
                profile_id="core-lite",
                display_name="FinSent Core Lite",
                model_class="FinSentNetCore",
                version="1.0.0-lite",
                price_window=20,
                price_features=16,
                text_tokens=32,
                output_heads=["direction_classification", "return_regression"],
                checkpoint_pattern="checkpoints/{ticker}_lite_model.pth",
                notes="Low-latency profile for constrained environments.",
            ),
        }
        self._active_profile_id = "core-v1"

    def set_active(self, profile_id: str) -> bool:
        if profile_id not in self._profiles:
            return False
        self._active_profile_id = profile_id
        return True

    def get_active(self) -> ModelProfile:
        return self._profiles[self._active_profile_id]

    def get_all(self) -> List[ModelProfile]:
        return list(self._profiles.values())

    def snapshot(self) -> Dict:
        return {
            "active_profile_id": self._active_profile_id,
            "active_profile": asdict(self.get_active()),
            "profiles": [asdict(p) for p in self.get_all()],
        }
