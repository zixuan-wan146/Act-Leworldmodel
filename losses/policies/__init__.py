"""Learned-controller training objectives."""

from losses.policies.gc_idm_regression import GCIDMRegressionObjective
from losses.policies.larc_rollout_consistency import LARCRolloutConsistencyObjective

__all__ = ["GCIDMRegressionObjective", "LARCRolloutConsistencyObjective"]
