"""
Generators module - Dataset subset generators.

Modules:
    base    - Base generator class
    stiff   - Stiff structures (compliance minimization)
    mech    - Mechanisms (displacement maximization)
    gold    - Gold standard (geometric NL re-simulation)
    path    - Path following (4-bar linkage seeding)
"""

from .stiff import (
    OptimizationConfig,
    OptimizationResult,
    optimize_compliance,
    generate_stiff_sample,
    generate_random_cantilever,
    generate_random_bridge,
)

from .mech import (
    MechConfig,
    MechProblem,
    optimize_mechanism,
    generate_mechanism_sample,
    generate_inverter_problem,
    generate_gripper_problem,
    generate_random_mechanism,
    MECH_GENERATORS,
)

__all__ = [
    # Stiff
    "OptimizationConfig",
    "OptimizationResult",
    "optimize_compliance",
    "generate_stiff_sample",
    "generate_random_cantilever",
    "generate_random_bridge",
    # Mech
    "MechConfig",
    "MechProblem",
    "optimize_mechanism",
    "generate_mechanism_sample",
    "generate_inverter_problem",
    "generate_gripper_problem",
    "generate_random_mechanism",
    "MECH_GENERATORS",
]

