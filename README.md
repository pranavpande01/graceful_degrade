# Abstract Memory Dynamics Framework (AMDF)

A framework for dynamic, long-term memory in AI systems with graceful forgetting.

## Overview

AMDF models how information is stored, forgotten, reinforced, and reconstructed over time under capacity constraints. Unlike static memory systems that either store everything (inefficient) or delete aggressively (catastrophic forgetting), AMDF introduces **graceful forgetting**: memory fades progressively at different rates for different kinds of information.

## Core Concepts

### Hierarchical Memory Layers

Each memory chunk has three layers with different decay rates:

| Layer | Content | Decay Rate |
|-------|---------|------------|
| **Core** | Semantic essence ("what it IS") | Slowest (10x) |
| **Detail** | Descriptive attributes ("what it's LIKE") | Medium (3x) |
| **Fine** | Exact surface form ("the EXACT content") | Fastest (1x) |

### Memory Dynamics

```
λ(t + Δt) = f_decay(λ(t), Δt) + f_excite(E(t))
```

- **Intensity** (λ): Controls accessibility and fidelity (0.0 → 1.0)
- **Decay**: Exponential, power-law, or step functions
- **Excitation**: Access reinforcement + cross-excitation from related memories

### Capacity Management

Memory is bounded by a global budget. When exceeded:
- Weakest layers retire first (fine → detail → core)
- Retired content can be reconstructed from surviving layers

## Installation

```bash
git clone https://github.com/pranavpande01/graceful_degrade
cd amdf
python3 demo.py
```

No dependencies required (stdlib only).

## Quick Start

```python
from memory_chunk import MemoryChunk
from memory_store import MemoryStore, MemoryStoreConfig
from decay_kernels import ExponentialDecay

# Create a memory store
store = MemoryStore(
    config=MemoryStoreConfig(capacity_budget=50.0),
    decay_kernel=ExponentialDecay(base_half_life=3600.0)
)

# Add a memory
chunk = MemoryChunk.from_text(
    text="The Eiffel Tower was built in 1889 in Paris.",
    core_summary="Eiffel Tower: Paris landmark, 1889",
    detail_summary="Iron tower built 1889 in Paris, France",
    source="wikipedia"
)
chunk_id = store.add(chunk)

# Recall reinforces the memory
store.recall(chunk_id)

# Apply decay over time
store.apply_decay(delta_t=3600.0)  # 1 hour

# Check what survives
content, layer, is_reconstructed = chunk.get_best_available_content()
```

## Project Structure

```
├── memory_chunk.py    # MemoryChunk and MemoryLayer classes
├── decay_kernels.py   # Exponential, power-law, step decay functions
├── memory_store.py    # Central store with capacity management
├── demo.py            # Interactive demos of all features
└── __init__.py        # Package exports
```

## Decay Kernels

| Kernel | Formula | Best For |
|--------|---------|----------|
| **Exponential** | λ₀ · e^(-kt) | Classic Ebbinghaus forgetting |
| **Power-law** | λ₀ / (1 + t/τ)^α | Long-term memory (slower decay) |
| **Step** | Discrete drops at intervals | Tiered storage systems |

## Features

- **Hierarchical layers** with differential decay rates
- **Spaced repetition effect**: access count slows decay
- **Cross-excitation**: recalling one memory reinforces related ones
- **Capacity enforcement**: automatic retirement under budget pressure
- **Reconstruction tracking**: marks regenerated content as approximate

## Demo

Run the interactive demo to see all features:

```bash
python3 demo.py
```

Output includes:
1. Basic memory creation
2. Decay dynamics over time
3. Access reinforcement
4. Cross-excitation between related memories
5. Capacity management
6. Graceful degradation scenarios
7. Decay kernel comparison

## Roadmap

- [ ] Embedding-based similarity (sentence-transformers)
- [ ] LLM-based layer generation (auto-summarization)
- [ ] LLM-based reconstruction for retired layers
- [ ] Persistent storage backend (SQLite + vector DB)
- [ ] Evaluation metrics for graceful degradation
