from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import threading
from collections import defaultdict

from memory_chunk import MemoryChunk, LayerType
from decay_kernels import DecayKernel, ExponentialDecay


@dataclass
class MemoryStoreConfig:
    capacity_budget: float = 100.0
    layer_weights: Dict[LayerType, float] = field(default_factory=lambda: {
        LayerType.CORE: 1.0,
        LayerType.DETAIL: 0.5,
        LayerType.FINE: 0.3
    })
    retirement_threshold: float = 0.01
    access_boost: float = 0.15
    excitation_strength: float = 0.3
    decay_interval: float = 60.0
    auto_decay: bool = False


class MemoryStore:
    def __init__(
        self,
        config: Optional[MemoryStoreConfig] = None,
        decay_kernel: Optional[DecayKernel] = None
    ):
        self.config = config or MemoryStoreConfig()
        self.decay_kernel = decay_kernel or ExponentialDecay()
        self._chunks: Dict[str, MemoryChunk] = {}
        self._similarity_graph: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._last_decay_time: float = time.time()
        self._decay_thread: Optional[threading.Thread] = None
        self._stop_decay = threading.Event()

        if self.config.auto_decay:
            self.start_background_decay()

    def add(self, chunk: MemoryChunk) -> str:
        self._chunks[chunk.chunk_id] = chunk
        self._enforce_capacity()
        return chunk.chunk_id

    def get(self, chunk_id: str) -> Optional[MemoryChunk]:
        return self._chunks.get(chunk_id)

    def recall(self, chunk_id: str) -> Optional[MemoryChunk]:
        chunk = self._chunks.get(chunk_id)
        if not chunk:
            return None
        chunk.access_all(self.config.access_boost)
        self._cross_excite(chunk_id)
        return chunk

    def remove(self, chunk_id: str) -> bool:
        if chunk_id in self._chunks:
            del self._chunks[chunk_id]
            if chunk_id in self._similarity_graph:
                del self._similarity_graph[chunk_id]
            for other_id in self._similarity_graph:
                self._similarity_graph[other_id].pop(chunk_id, None)
            return True
        return False

    def list_chunks(self) -> List[MemoryChunk]:
        return list(self._chunks.values())

    def connect(self, chunk_id_a: str, chunk_id_b: str, similarity: float = 1.0) -> None:
        if chunk_id_a in self._chunks and chunk_id_b in self._chunks:
            self._similarity_graph[chunk_id_a][chunk_id_b] = similarity
            self._similarity_graph[chunk_id_b][chunk_id_a] = similarity

    def get_connections(self, chunk_id: str) -> Dict[str, float]:
        return dict(self._similarity_graph.get(chunk_id, {}))

    def _cross_excite(self, source_chunk_id: str) -> None:
        connections = self._similarity_graph.get(source_chunk_id, {})
        for target_id, similarity in connections.items():
            target_chunk = self._chunks.get(target_id)
            if target_chunk:
                boost = self.config.access_boost * similarity * self.config.excitation_strength
                target_chunk.access_all(boost)

    def apply_decay(self, delta_t: Optional[float] = None) -> Dict[str, any]:
        now = time.time()
        if delta_t is None:
            delta_t = now - self._last_decay_time
        self._last_decay_time = now

        stats = {"chunks_processed": 0, "layers_decayed": 0, "layers_retired": 0}

        for chunk in self._chunks.values():
            stats["chunks_processed"] += 1
            for layer_type in LayerType:
                layer = chunk.get_layer(layer_type)
                if layer and not layer.is_retired(self.config.retirement_threshold):
                    new_intensity = self.decay_kernel.decay(
                        intensity=layer.intensity,
                        delta_t=delta_t,
                        layer_type=layer_type,
                        access_count=layer.access_count
                    )
                    layer.intensity = new_intensity
                    stats["layers_decayed"] += 1
                    if layer.is_retired(self.config.retirement_threshold):
                        stats["layers_retired"] += 1
        return stats

    def start_background_decay(self) -> None:
        if self._decay_thread is not None:
            return
        self._stop_decay.clear()

        def decay_loop():
            while not self._stop_decay.is_set():
                self.apply_decay()
                self._stop_decay.wait(self.config.decay_interval)

        self._decay_thread = threading.Thread(target=decay_loop, daemon=True)
        self._decay_thread.start()

    def stop_background_decay(self) -> None:
        if self._decay_thread is not None:
            self._stop_decay.set()
            self._decay_thread.join(timeout=5.0)
            self._decay_thread = None

    def current_usage(self) -> float:
        total = 0.0
        for chunk in self._chunks.values():
            total += chunk.total_intensity(self.config.layer_weights)
        return total

    def usage_ratio(self) -> float:
        return self.current_usage() / self.config.capacity_budget

    def _enforce_capacity(self) -> int:
        demoted = 0
        usage = self.current_usage()

        while usage > self.config.capacity_budget:
            lowest_layer = None
            lowest_intensity = float('inf')

            for chunk in self._chunks.values():
                for layer_type in [LayerType.FINE, LayerType.DETAIL, LayerType.CORE]:
                    layer = chunk.get_layer(layer_type)
                    if layer and not layer.is_retired(self.config.retirement_threshold):
                        if layer.intensity < lowest_intensity:
                            lowest_intensity = layer.intensity
                            lowest_layer = layer

            if lowest_layer is None:
                break

            lowest_layer.intensity = 0.0
            demoted += 1
            usage = self.current_usage()

        return demoted

    def search_by_intensity(
        self,
        min_intensity: float = 0.1,
        layer_type: Optional[LayerType] = None
    ) -> List[MemoryChunk]:
        results = []
        for chunk in self._chunks.values():
            if layer_type:
                layer = chunk.get_layer(layer_type)
                if layer and layer.intensity >= min_intensity:
                    results.append(chunk)
            else:
                if chunk.total_intensity() >= min_intensity:
                    results.append(chunk)

        results.sort(
            key=lambda c: c.total_intensity() if not layer_type
                else (c.get_layer(layer_type).intensity if c.get_layer(layer_type) else 0),
            reverse=True
        )
        return results

    def get_strongest_memories(self, n: int = 10) -> List[MemoryChunk]:
        chunks = list(self._chunks.values())
        chunks.sort(key=lambda c: c.total_intensity(), reverse=True)
        return chunks[:n]

    def get_weakest_memories(self, n: int = 10) -> List[MemoryChunk]:
        chunks = list(self._chunks.values())
        chunks.sort(key=lambda c: c.total_intensity())
        return chunks[:n]

    def stats(self) -> Dict[str, any]:
        total_chunks = len(self._chunks)
        active_layers = {lt: 0 for lt in LayerType}
        retired_layers = {lt: 0 for lt in LayerType}

        for chunk in self._chunks.values():
            for layer_type in LayerType:
                layer = chunk.get_layer(layer_type)
                if layer:
                    if layer.is_retired(self.config.retirement_threshold):
                        retired_layers[layer_type] += 1
                    else:
                        active_layers[layer_type] += 1

        return {
            "total_chunks": total_chunks,
            "capacity_usage": self.current_usage(),
            "capacity_budget": self.config.capacity_budget,
            "usage_ratio": self.usage_ratio(),
            "active_layers": {k.value: v for k, v in active_layers.items()},
            "retired_layers": {k.value: v for k, v in retired_layers.items()},
            "decay_kernel": self.decay_kernel.name()
        }

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"MemoryStore(chunks={stats['total_chunks']}, "
            f"usage={stats['usage_ratio']:.1%}, "
            f"kernel={stats['decay_kernel']})"
        )

    def __len__(self) -> int:
        return len(self._chunks)
