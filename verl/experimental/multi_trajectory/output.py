"""
Output data structures for multi-trajectory support.
"""

from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel


class TrajectoryOutput(BaseModel):
    """Output from a single trajectory within a multi-trajectory group."""

    prompt_ids: list[int]
    """Prompt token ids for this trajectory."""

    response_ids: list[int]
    """Response token ids for this trajectory."""

    response_mask: list[int]
    """Response mask: 1 for LLM generated tokens, 0 for tool/context tokens."""

    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""

    trajectory_id: str = ""
    """Unique identifier for this trajectory within the group."""

    role: str = "worker"
    """Role of this trajectory: 'worker', 'aggregator', etc."""

    is_reward_source: bool = False
    """Whether this trajectory's final answer determines the group's reward."""

    extra_fields: dict[str, Any] = {}
    """Additional fields for this trajectory."""


class MultiTrajectoryOutput(BaseModel):
    """Output containing multiple related trajectories that share a reward.

    When an agent spawns child trajectories, all trajectories in this group
    will receive the same reward based on the trajectory marked as `is_reward_source`.
    """

    trajectories: list[TrajectoryOutput]
    """List of all trajectories in this group."""

    trajectory_group_id: str
    """Unique identifier linking all trajectories that share a reward."""

    reward_source_index: int = -1
    """Index of the trajectory whose answer determines the reward. -1 means last."""

    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data shared across trajectories."""

    num_turns: int = 0
    """Total number of chat turns across all trajectories."""

    metrics: dict[str, Any] = {}
    """Performance metrics for the trajectory group."""

    extra_fields: dict[str, Any] = {}
    """Additional fields for the group."""

    def get_reward_source_trajectory(self) -> TrajectoryOutput:
        """Get the trajectory that determines the group's reward."""
        if self.reward_source_index >= 0:
            return self.trajectories[self.reward_source_index]
        # Find the one marked as reward source
        for traj in self.trajectories:
            if traj.is_reward_source:
                return traj
        # Default to last trajectory
        return self.trajectories[-1]

    def mark_reward_source(self):
        """Mark the reward source trajectory and ensure only one is marked."""
        reward_idx = (
            self.reward_source_index
            if self.reward_source_index >= 0
            else len(self.trajectories) - 1
        )
        for i, traj in enumerate(self.trajectories):
            traj.is_reward_source = i == reward_idx


@dataclass
class SpawnedTrajectoryHandle:
    """Handle returned when spawning a trajectory, used to join later."""

    trajectory_id: str
    """Unique identifier for the spawned trajectory."""

    trajectory_group_id: str
    """Group ID linking related trajectories."""

    agent_loop: str = "tool_agent"
    """Agent loop type used for the spawned trajectory."""

    status: str = "pending"
    """Status: 'pending', 'running', 'completed', 'failed'."""

    result: Optional[TrajectoryOutput] = None
    """Result once the trajectory completes."""
