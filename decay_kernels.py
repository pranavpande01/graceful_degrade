from abc import ABC, abstractmethod
from typing import Dict, Optional
import math

from memory_chunk import LayerType


class DecayKernel(ABC):
    @abstractmethod
    def decay(
        self,
        intensity: float,
        delta_t: float,
        layer_type: LayerType,
        access_count: int = 0
    ) -> float:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class ExponentialDecay(DecayKernel):
    def __init__(
        self,
        base_half_life: float = 3600.0,
        layer_multipliers: Optional[Dict[LayerType, float]] = None
    ):
        self.base_half_life = base_half_life
        self.layer_multipliers = layer_multipliers or {
            LayerType.CORE: 10.0,
            LayerType.DETAIL: 3.0,
            LayerType.FINE: 1.0
        }

    def decay(
        self,
        intensity: float,
        delta_t: float,
        layer_type: LayerType,
        access_count: int = 0
    ) -> float:
        multiplier = self.layer_multipliers.get(layer_type, 1.0)
        access_bonus = 1.0 + (0.2 * min(access_count, 10))
        effective_half_life = self.base_half_life * multiplier * access_bonus
        k = math.log(2) / effective_half_life
        new_intensity = intensity * math.exp(-k * delta_t)
        return max(0.0, new_intensity)

    def name(self) -> str:
        return "exponential"


class PowerLawDecay(DecayKernel):
    def __init__(
        self,
        tau: float = 3600.0,
        alpha: float = 0.5,
        layer_multipliers: Optional[Dict[LayerType, float]] = None
    ):
        self.tau = tau
        self.alpha = alpha
        self.layer_multipliers = layer_multipliers or {
            LayerType.CORE: 10.0,
            LayerType.DETAIL: 3.0,
            LayerType.FINE: 1.0
        }

    def decay(
        self,
        intensity: float,
        delta_t: float,
        layer_type: LayerType,
        access_count: int = 0
    ) -> float:
        multiplier = self.layer_multipliers.get(layer_type, 1.0)
        access_bonus = 1.0 + (0.2 * min(access_count, 10))
        effective_tau = self.tau * multiplier * access_bonus
        decay_factor = 1.0 / ((1.0 + delta_t / effective_tau) ** self.alpha)
        new_intensity = intensity * decay_factor
        return max(0.0, new_intensity)

    def name(self) -> str:
        return "power_law"


class StepDecay(DecayKernel):
    def __init__(
        self,
        step_interval: float = 3600.0,
        step_size: float = 0.2,
        layer_multipliers: Optional[Dict[LayerType, float]] = None
    ):
        self.step_interval = step_interval
        self.step_size = step_size
        self.layer_multipliers = layer_multipliers or {
            LayerType.CORE: 10.0,
            LayerType.DETAIL: 3.0,
            LayerType.FINE: 1.0
        }

    def decay(
        self,
        intensity: float,
        delta_t: float,
        layer_type: LayerType,
        access_count: int = 0
    ) -> float:
        multiplier = self.layer_multipliers.get(layer_type, 1.0)
        access_bonus = 1.0 + (0.2 * min(access_count, 10))
        effective_interval = self.step_interval * multiplier * access_bonus
        num_steps = int(delta_t / effective_interval)
        if num_steps == 0:
            return intensity
        new_intensity = intensity * ((1.0 - self.step_size) ** num_steps)
        return max(0.0, new_intensity)

    def name(self) -> str:
        return "step"


class CompositeDecay(DecayKernel):
    def __init__(self, kernels: Dict[LayerType, DecayKernel]):
        self.kernels = kernels
        self._default = ExponentialDecay()

    def decay(
        self,
        intensity: float,
        delta_t: float,
        layer_type: LayerType,
        access_count: int = 0
    ) -> float:
        kernel = self.kernels.get(layer_type, self._default)
        return kernel.decay(intensity, delta_t, layer_type, access_count)

    def name(self) -> str:
        return "composite"


def create_decay_kernel(kernel_type: str = "exponential", **kwargs) -> DecayKernel:
    kernels = {
        "exponential": ExponentialDecay,
        "power_law": PowerLawDecay,
        "step": StepDecay
    }
    if kernel_type not in kernels:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    return kernels[kernel_type](**kwargs)
