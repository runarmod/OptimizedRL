"""Dynamic Vehicle Scheduling Problem (DVSP).

Python port of the DVSP used in "Structured Reinforcement Learning for
Combinatorial Decision-Making" (Hoppe et al., NeurIPS 2025). The reference
implementation is DynamicVehicleRouting.jl (BatyLeo), commit 0af4301, which is
the version pinned in the Structured-RL repository's Manifest.toml.

All indices are 0-based here; index 0 is always the depot.
"""
