from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import time
import uuid


class LayerType(Enum):
    CORE = "core"
    DETAIL = "detail"
    FINE = "fine"


@dataclass
class MemoryLayer:
    layer_type: LayerType
    content: str
    embedding: Optional[list] = None
    intensity: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    is_reconstructed: bool = False

    def access(self, boost: float = 0.1) -> None:
        self.last_accessed = time.time()
        self.access_count += 1
        self.intensity = min(1.0, self.intensity + boost)

    def is_retired(self, threshold: float = 0.01) -> bool:
        return self.intensity < threshold


@dataclass
class MemoryChunk:
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    core: Optional[MemoryLayer] = None
    detail: Optional[MemoryLayer] = None
    fine: Optional[MemoryLayer] = None
    connections: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @classmethod
    def from_text(
        cls,
        text: str,
        core_summary: str,
        detail_summary: Optional[str] = None,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> "MemoryChunk":
        chunk = cls(source=source, metadata=metadata or {})
        chunk.fine = MemoryLayer(layer_type=LayerType.FINE, content=text, intensity=1.0)
        chunk.core = MemoryLayer(layer_type=LayerType.CORE, content=core_summary, intensity=1.0)
        if detail_summary:
            chunk.detail = MemoryLayer(layer_type=LayerType.DETAIL, content=detail_summary, intensity=1.0)
        return chunk

    def get_layer(self, layer_type: LayerType) -> Optional[MemoryLayer]:
        if layer_type == LayerType.CORE:
            return self.core
        elif layer_type == LayerType.DETAIL:
            return self.detail
        elif layer_type == LayerType.FINE:
            return self.fine
        return None

    def get_best_available_content(self) -> tuple[str, LayerType, bool]:
        if self.fine and not self.fine.is_retired():
            return self.fine.content, LayerType.FINE, self.fine.is_reconstructed
        if self.detail and not self.detail.is_retired():
            return self.detail.content, LayerType.DETAIL, self.detail.is_reconstructed
        if self.core:
            return self.core.content, LayerType.CORE, self.core.is_reconstructed
        return "", LayerType.CORE, False

    def total_intensity(self, weights: Optional[Dict[LayerType, float]] = None) -> float:
        if weights is None:
            weights = {LayerType.CORE: 1.0, LayerType.DETAIL: 0.5, LayerType.FINE: 0.3}
        total = 0.0
        for layer_type, weight in weights.items():
            layer = self.get_layer(layer_type)
            if layer:
                total += weight * layer.intensity
        return total

    def access_all(self, boost: float = 0.1) -> None:
        for layer in [self.core, self.detail, self.fine]:
            if layer and not layer.is_retired():
                layer.access(boost)

    def get_active_layers(self, threshold: float = 0.01) -> list[LayerType]:
        active = []
        for layer_type in LayerType:
            layer = self.get_layer(layer_type)
            if layer and not layer.is_retired(threshold):
                active.append(layer_type)
        return active

    def __repr__(self) -> str:
        layers = self.get_active_layers()
        layer_str = ", ".join(l.value for l in layers)
        core_preview = self.core.content[:50] if self.core else "N/A"
        return f"MemoryChunk({self.chunk_id[:8]}..., layers=[{layer_str}], core='{core_preview}...')"
