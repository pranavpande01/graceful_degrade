#!/usr/bin/env python3
"""
AMDF Prototype Demo

This script demonstrates the core mechanics of the Abstract Memory Dynamics Framework:
1. Creating memory chunks with hierarchical layers
2. Decay dynamics over time
3. Cross-excitation between related memories
4. Capacity management
5. Memory retrieval and reconstruction scenarios
"""

import time
from memory_chunk import MemoryChunk, LayerType
from decay_kernels import ExponentialDecay, PowerLawDecay, create_decay_kernel
from memory_store import MemoryStore, MemoryStoreConfig


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def print_chunk_status(chunk: MemoryChunk, indent: str = "") -> None:
    """Print detailed status of a memory chunk."""
    print(f"{indent}Chunk: {chunk.chunk_id[:8]}...")
    print(f"{indent}  Source: {chunk.source}")

    for layer_type in LayerType:
        layer = chunk.get_layer(layer_type)
        if layer:
            status = "RETIRED" if layer.is_retired() else "active"
            recon = " (reconstructed)" if layer.is_reconstructed else ""
            print(f"{indent}  {layer_type.value:8}: intensity={layer.intensity:.4f} "
                  f"access_count={layer.access_count} [{status}]{recon}")
            print(f"{indent}            content: '{layer.content[:50]}...'")


def demo_basic_memory_creation():
    """Demo 1: Creating memory chunks with hierarchical layers."""
    print_header("Demo 1: Basic Memory Creation")

    # Create a memory chunk from text
    chunk = MemoryChunk.from_text(
        text="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.",
        core_summary="Eiffel Tower: famous iron tower in Paris, built 1889",
        detail_summary="Wrought-iron lattice tower on Champ de Mars, Paris. Named after Gustave Eiffel, built 1887-1889.",
        source="Wikipedia",
        metadata={"topic": "landmarks", "location": "Paris"}
    )

    print("Created a memory chunk about the Eiffel Tower:")
    print_chunk_status(chunk)

    print("\n• Core layer = semantic essence (slow decay)")
    print("• Detail layer = descriptive attributes (medium decay)")
    print("• Fine layer = exact text (fast decay)")

    return chunk


def demo_decay_dynamics():
    """Demo 2: Decay dynamics over time."""
    print_header("Demo 2: Decay Dynamics")

    # Create a fresh chunk
    chunk = MemoryChunk.from_text(
        text="Python is a high-level programming language known for its readability.",
        core_summary="Python: readable high-level programming language",
        detail_summary="High-level language emphasizing code readability",
        source="demo"
    )

    # Create decay kernel with fast decay for demo (1 minute half-life)
    decay = ExponentialDecay(base_half_life=60.0)  # 1 minute

    print("Initial state:")
    print_chunk_status(chunk, "  ")

    # Simulate 5 minutes of decay
    print("\nSimulating 5 time steps (1 minute each):")
    for step in range(1, 6):
        # Apply decay to each layer
        for layer_type in LayerType:
            layer = chunk.get_layer(layer_type)
            if layer:
                layer.intensity = decay.decay(
                    intensity=layer.intensity,
                    delta_t=60.0,  # 1 minute
                    layer_type=layer_type,
                    access_count=layer.access_count
                )

        print(f"\n  After {step} minute(s):")
        for lt in LayerType:
            layer = chunk.get_layer(lt)
            if layer:
                status = "RETIRED" if layer.is_retired() else "active"
                print(f"    {lt.value:8}: {layer.intensity:.4f} [{status}]")

    print("\nNote: Fine layer decays fastest, Core layer decays slowest")


def demo_access_reinforcement():
    """Demo 3: Memory reinforcement through access."""
    print_header("Demo 3: Access Reinforcement")

    # Create store with fast decay for demo
    config = MemoryStoreConfig(capacity_budget=50.0)
    decay = ExponentialDecay(base_half_life=30.0)  # 30 seconds
    store = MemoryStore(config=config, decay_kernel=decay)

    # Add a memory
    chunk = MemoryChunk.from_text(
        text="Machine learning is a subset of artificial intelligence.",
        core_summary="ML: subset of AI",
        detail_summary="Machine learning is part of AI field",
        source="demo"
    )
    chunk_id = store.add(chunk)

    print("Initial state:")
    print(f"  Core intensity: {chunk.core.intensity:.4f}")
    print(f"  Access count: {chunk.core.access_count}")

    # Simulate decay
    print("\nApplying 2 minutes of decay...")
    store.apply_decay(delta_t=120.0)
    print(f"  Core intensity after decay: {chunk.core.intensity:.4f}")

    # Now recall (access) the memory
    print("\nRecalling the memory (triggering reinforcement)...")
    store.recall(chunk_id)
    print(f"  Core intensity after recall: {chunk.core.intensity:.4f}")
    print(f"  Access count: {chunk.core.access_count}")

    # More decay
    print("\nApplying 2 more minutes of decay...")
    store.apply_decay(delta_t=120.0)
    print(f"  Core intensity: {chunk.core.intensity:.4f}")
    print("  (Decays slower now due to access count!)")


def demo_cross_excitation():
    """Demo 4: Cross-excitation between related memories."""
    print_header("Demo 4: Cross-Excitation")

    config = MemoryStoreConfig(
        capacity_budget=100.0,
        excitation_strength=0.5  # Strong cross-excitation
    )
    store = MemoryStore(config=config)

    # Create related memories
    mona_lisa = MemoryChunk.from_text(
        text="The Mona Lisa is a portrait painting by Leonardo da Vinci.",
        core_summary="Mona Lisa: painting by da Vinci",
        source="art"
    )

    da_vinci = MemoryChunk.from_text(
        text="Leonardo da Vinci was an Italian polymath of the Renaissance.",
        core_summary="Da Vinci: Renaissance polymath",
        source="biography"
    )

    renaissance = MemoryChunk.from_text(
        text="The Renaissance was a period of cultural rebirth in Europe.",
        core_summary="Renaissance: European cultural rebirth",
        source="history"
    )

    # Add to store
    ml_id = store.add(mona_lisa)
    dv_id = store.add(da_vinci)
    ren_id = store.add(renaissance)

    # Create semantic connections
    store.connect(ml_id, dv_id, similarity=0.9)   # Strong: same artist
    store.connect(dv_id, ren_id, similarity=0.7)  # Medium: same era
    store.connect(ml_id, ren_id, similarity=0.5)  # Weaker: indirect

    print("Created 3 related memories with connections:")
    print("  Mona Lisa <--0.9--> Da Vinci <--0.7--> Renaissance")
    print("  Mona Lisa <----------0.5-----------> Renaissance")

    # Apply some decay
    print("\nApplying decay to all memories...")
    store.apply_decay(delta_t=300.0)

    print("\nIntensities before recall:")
    print(f"  Mona Lisa:   {mona_lisa.core.intensity:.4f}")
    print(f"  Da Vinci:    {da_vinci.core.intensity:.4f}")
    print(f"  Renaissance: {renaissance.core.intensity:.4f}")

    # Recall just the Mona Lisa
    print("\nRecalling 'Mona Lisa' only...")
    store.recall(ml_id)

    print("\nIntensities after recall (note cross-excitation!):")
    print(f"  Mona Lisa:   {mona_lisa.core.intensity:.4f} (directly accessed)")
    print(f"  Da Vinci:    {da_vinci.core.intensity:.4f} (strongly excited)")
    print(f"  Renaissance: {renaissance.core.intensity:.4f} (weakly excited)")


def demo_capacity_management():
    """Demo 5: Capacity constraints and layer retirement."""
    print_header("Demo 5: Capacity Management")

    # Small capacity to force eviction
    config = MemoryStoreConfig(capacity_budget=5.0)
    store = MemoryStore(config=config)

    print(f"Memory budget: {config.capacity_budget}")
    print("Adding memories until capacity is exceeded...\n")

    topics = [
        ("Python programming language", "Python: programming language"),
        ("JavaScript for web development", "JavaScript: web language"),
        ("Rust for systems programming", "Rust: systems language"),
        ("Go for cloud services", "Go: cloud language"),
        ("TypeScript adds types to JS", "TypeScript: typed JS"),
    ]

    for i, (text, core) in enumerate(topics):
        chunk = MemoryChunk.from_text(text=text, core_summary=core, source=f"topic_{i}")
        store.add(chunk)

        stats = store.stats()
        print(f"Added '{core[:30]}...'")
        print(f"  Usage: {stats['usage_ratio']:.1%} of budget")
        print(f"  Retired layers: {sum(stats['retired_layers'].values())}")

    print("\nFinal store status:")
    print(f"  Total chunks: {len(store)}")
    print(f"  Capacity usage: {store.usage_ratio():.1%}")

    stats = store.stats()
    print(f"  Active layers by type:")
    for layer_type, count in stats['active_layers'].items():
        retired = stats['retired_layers'][layer_type]
        print(f"    {layer_type}: {count} active, {retired} retired")


def demo_memory_reconstruction_scenario():
    """Demo 6: Graceful degradation and reconstruction scenario."""
    print_header("Demo 6: Graceful Degradation")

    chunk = MemoryChunk.from_text(
        text="On July 20, 1969, Neil Armstrong became the first human to walk on the Moon, saying 'That's one small step for man, one giant leap for mankind.'",
        core_summary="1969 Moon landing: Armstrong first to walk on Moon",
        detail_summary="Neil Armstrong walked on Moon July 20, 1969. Famous quote about small step/giant leap.",
        source="history"
    )

    decay = ExponentialDecay(base_half_life=30.0)

    print("Original memory with all layers:")
    print_chunk_status(chunk, "  ")

    # Simulate aggressive decay (like 10 minutes passing)
    print("\nSimulating significant time passage...")
    for _ in range(10):
        for layer_type in LayerType:
            layer = chunk.get_layer(layer_type)
            if layer:
                layer.intensity = decay.decay(
                    layer.intensity, 60.0, layer_type, layer.access_count
                )

    print("\nAfter decay:")
    print_chunk_status(chunk, "  ")

    # Get best available content
    content, layer_type, is_reconstructed = chunk.get_best_available_content()
    print(f"\nBest available content (from {layer_type.value} layer):")
    print(f"  '{content}'")

    if layer_type != LayerType.FINE:
        print("\n  ⚠ Fine details have been forgotten!")
        print("  The exact quote is no longer available.")
        print("  A generative model could RECONSTRUCT approximate details,")
        print("  but they would be marked as 'is_reconstructed=True'")


def demo_different_decay_kernels():
    """Demo 7: Comparing different decay kernels."""
    print_header("Demo 7: Decay Kernel Comparison")

    # Same chunk, different decay functions
    kernels = [
        ("Exponential", ExponentialDecay(base_half_life=60.0)),
        ("Power-law", create_decay_kernel("power_law", tau=60.0, alpha=0.5)),
        ("Step", create_decay_kernel("step", step_interval=60.0, step_size=0.3)),
    ]

    print("Comparing decay of FINE layer over 5 minutes:\n")
    print("Time (min)  | Exponential | Power-law  | Step")
    print("-" * 55)

    results = {name: [1.0] for name, _ in kernels}

    for minute in range(6):
        row = [f"    {minute}       "]
        for name, kernel in kernels:
            intensity = results[name][-1]
            if minute > 0:
                intensity = kernel.decay(
                    intensity=intensity,
                    delta_t=60.0,
                    layer_type=LayerType.FINE,
                    access_count=0
                )
                results[name].append(intensity)
            row.append(f"{intensity:.4f}      ")
        print("|".join(row))

    print("\nNote:")
    print("  • Exponential: Smooth continuous decay")
    print("  • Power-law: Slower long-term decay (better for LTM)")
    print("  • Step: Discrete drops (good for tiered storage)")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("   ABSTRACT MEMORY DYNAMICS FRAMEWORK (AMDF)")
    print("   Minimal Prototype Demo")
    print("=" * 60)

    demo_basic_memory_creation()
    demo_decay_dynamics()
    demo_access_reinforcement()
    demo_cross_excitation()
    demo_capacity_management()
    demo_memory_reconstruction_scenario()
    demo_different_decay_kernels()

    print_header("Demo Complete!")
    print("This prototype demonstrates the core AMDF mechanics:")
    print("  ✓ Hierarchical memory layers (core/detail/fine)")
    print("  ✓ Configurable decay kernels")
    print("  ✓ Access-based reinforcement")
    print("  ✓ Cross-excitation between related memories")
    print("  ✓ Capacity management with graceful retirement")
    print("  ✓ Graceful degradation scenarios")
    print()
    print("Next steps for a full implementation:")
    print("  • Add embedding-based similarity computation")
    print("  • Implement LLM-based layer generation (summarization)")
    print("  • Add LLM-based reconstruction for retired layers")
    print("  • Persistent storage backend (SQLite, vector DB)")
    print("  • Evaluation metrics for graceful degradation")
    print()


if __name__ == "__main__":
    main()
