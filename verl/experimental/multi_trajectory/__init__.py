"""
Multi-trajectory support for verl.

This module enables spawning multiple trainable trajectories from a single prompt,
where all trajectories in a group share the same final reward.
"""

from .integration import (
    expand_multi_trajectory_batch,
    get_multi_trajectory_metrics,
    postprocess_multi_trajectory_rewards,
)
from .output import MultiTrajectoryOutput, TrajectoryOutput
from .reward_broadcast import (
    add_trajectory_group_fields,
    broadcast_reward_by_trajectory_group,
    compute_trajectory_group_metrics,
)
from .tools import JoinTrajectoriesTool, SpawnTrajectoryTool
from .trajectory_manager import TrajectoryManager

__all__ = [
    "TrajectoryManager",
    "SpawnTrajectoryTool",
    "JoinTrajectoriesTool",
    "broadcast_reward_by_trajectory_group",
    "add_trajectory_group_fields",
    "compute_trajectory_group_metrics",
    "MultiTrajectoryOutput",
    "TrajectoryOutput",
    "expand_multi_trajectory_batch",
    "postprocess_multi_trajectory_rewards",
    "get_multi_trajectory_metrics",
]
