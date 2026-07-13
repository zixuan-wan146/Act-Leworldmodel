"""World-model and learned-controller objectives."""

from losses.policies import GCIDMRegressionObjective, LARCRolloutConsistencyObjective
from losses.world_model import DensePrefixObjective

__all__ = [
    "DensePrefixObjective",
    "GCIDMRegressionObjective",
    "LARCRolloutConsistencyObjective",
]
