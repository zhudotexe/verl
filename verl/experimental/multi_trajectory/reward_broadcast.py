"""
Reward broadcasting for multi-trajectory training.

When multiple trajectories share a trajectory_group_id, they should all
receive the same reward (computed from the reward source trajectory).
This module provides utilities to broadcast rewards across trajectory groups.
"""

import logging
from typing import Optional

import numpy as np

from verl.protocol import DataProto

logger = logging.getLogger(__name__)


def broadcast_reward_by_trajectory_group(
    batch: DataProto,
    reward_key: str = "rm_scores",
    group_id_key: str = "trajectory_group_id",
    reward_source_key: str = "is_reward_source",
) -> DataProto:
    """Broadcast rewards within trajectory groups.

    For trajectories that share the same trajectory_group_id, this function
    copies the reward from the trajectory marked as is_reward_source to all
    other trajectories in the group.

    Args:
        batch: DataProto containing the batch data.
        reward_key: Key in batch.batch for the reward tensor.
        group_id_key: Key in batch.non_tensor_batch for trajectory group IDs.
        reward_source_key: Key in batch.non_tensor_batch for is_reward_source flags.

    Returns:
        DataProto: Updated batch with broadcasted rewards.
    """
    # Check if multi-trajectory fields exist
    if group_id_key not in batch.non_tensor_batch:
        logger.debug(f"No {group_id_key} in batch, skipping reward broadcast")
        return batch

    if reward_key not in batch.batch.keys():
        logger.warning(f"No {reward_key} in batch, skipping reward broadcast")
        return batch

    group_ids = batch.non_tensor_batch[group_id_key]
    reward_tensor = batch.batch[reward_key]
    is_reward_source = batch.non_tensor_batch.get(reward_source_key)

    # Build mapping from group_id to (reward_source_idx, member_indices)
    group_to_members: dict[str, list[int]] = {}
    group_to_source: dict[str, int] = {}

    for i, gid in enumerate(group_ids):
        if gid is None:
            # Not part of a multi-trajectory group
            continue

        gid_str = str(gid)
        if gid_str not in group_to_members:
            group_to_members[gid_str] = []
        group_to_members[gid_str].append(i)

        # Track reward source
        if is_reward_source is not None and is_reward_source[i]:
            group_to_source[gid_str] = i

    # For groups without explicit reward source, use the last member
    for gid, members in group_to_members.items():
        if gid not in group_to_source:
            group_to_source[gid] = members[-1]

    # Broadcast rewards
    for gid, members in group_to_members.items():
        if len(members) <= 1:
            continue  # No broadcast needed for single-member groups

        source_idx = group_to_source[gid]
        source_reward = reward_tensor[source_idx]

        for member_idx in members:
            if member_idx != source_idx:
                # Copy the reward from source to this member
                reward_tensor[member_idx] = source_reward.clone()
                logger.debug(
                    f"Broadcast reward from trajectory {source_idx} to {member_idx} "
                    f"in group {gid}"
                )

    batch.batch[reward_key] = reward_tensor
    return batch


def add_trajectory_group_fields(
    batch: DataProto,
    trajectory_group_ids: Optional[np.ndarray] = None,
    is_reward_source: Optional[np.ndarray] = None,
) -> DataProto:
    """Add trajectory group fields to a batch.

    This is a helper to add the required fields for multi-trajectory training
    to an existing batch.

    Args:
        batch: DataProto to modify.
        trajectory_group_ids: Array of group IDs (None for non-grouped trajectories).
        is_reward_source: Array of booleans indicating reward source trajectories.

    Returns:
        DataProto: Updated batch with trajectory group fields.
    """
    batch_size = len(batch)

    if trajectory_group_ids is None:
        trajectory_group_ids = np.array([None] * batch_size, dtype=object)

    if is_reward_source is None:
        is_reward_source = np.array([False] * batch_size, dtype=bool)

    batch.non_tensor_batch["trajectory_group_id"] = trajectory_group_ids
    batch.non_tensor_batch["is_reward_source"] = is_reward_source

    return batch


def compute_trajectory_group_metrics(batch: DataProto) -> dict[str, float]:
    """Compute metrics related to trajectory groups.

    Args:
        batch: DataProto containing trajectory group information.

    Returns:
        dict: Metrics including group counts, average group sizes, etc.
    """
    metrics = {}

    group_id_key = "trajectory_group_id"
    if group_id_key not in batch.non_tensor_batch:
        return metrics

    group_ids = batch.non_tensor_batch[group_id_key]

    # Count groups and sizes
    group_counts: dict[str, int] = {}
    ungrouped_count = 0

    for gid in group_ids:
        if gid is None:
            ungrouped_count += 1
        else:
            gid_str = str(gid)
            group_counts[gid_str] = group_counts.get(gid_str, 0) + 1

    metrics["multi_traj/num_groups"] = len(group_counts)
    metrics["multi_traj/ungrouped_trajectories"] = ungrouped_count
    metrics["multi_traj/grouped_trajectories"] = len(group_ids) - ungrouped_count

    if group_counts:
        sizes = list(group_counts.values())
        metrics["multi_traj/avg_group_size"] = sum(sizes) / len(sizes)
        metrics["multi_traj/max_group_size"] = max(sizes)
        metrics["multi_traj/min_group_size"] = min(sizes)

    return metrics
