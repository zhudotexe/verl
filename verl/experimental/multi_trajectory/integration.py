"""
Integration utilities for multi-trajectory training.

This module provides functions to integrate multi-trajectory support into
the existing verl training pipeline.
"""

import logging
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from verl.protocol import DataProto
from .output import MultiTrajectoryOutput, TrajectoryOutput
from .reward_broadcast import (
    add_trajectory_group_fields,
    broadcast_reward_by_trajectory_group,
    compute_trajectory_group_metrics,
)

logger = logging.getLogger(__name__)


def expand_multi_trajectory_batch(
    batch: DataProto,
    tokenizer: Any,
    prompt_length: int,
    response_length: int,
) -> DataProto:
    """Expand a batch containing multi-trajectory outputs into individual trajectories.

    When an AgentLoopWorker returns outputs with spawned trajectories, this function
    expands them so each trajectory becomes a separate row in the batch. All trajectories
    in a group share the same trajectory_group_id.

    Args:
        batch: DataProto containing outputs, some of which may have multi_trajectory_output.
        tokenizer: Tokenizer for padding.
        prompt_length: Max prompt length for padding.
        response_length: Max response length for padding.

    Returns:
        DataProto: Expanded batch where each trajectory is a separate row.
    """
    # Check if we have any multi-trajectory outputs
    extra_fields = batch.non_tensor_batch.get("tool_extra_fields")
    if extra_fields is None:
        # No extra fields, nothing to expand
        return add_trajectory_group_fields(batch)

    has_multi_traj = any(
        ef.get("multi_trajectory_output") is not None
        for ef in extra_fields
        if ef is not None
    )

    if not has_multi_traj:
        # No multi-trajectory outputs, just add the fields with None values
        return add_trajectory_group_fields(batch)

    # We need to expand the batch
    expanded_tensors = {key: [] for key in batch.batch.keys()}
    expanded_non_tensors = {key: [] for key in batch.non_tensor_batch.keys()}
    expanded_non_tensors["trajectory_group_id"] = []
    expanded_non_tensors["is_reward_source"] = []

    for i in range(len(batch)):
        item = batch[i]
        extra = extra_fields[i] if extra_fields[i] is not None else {}
        multi_traj = extra.get("multi_trajectory_output")

        if multi_traj is None:
            # Regular trajectory, just copy it
            for key in batch.batch.keys():
                expanded_tensors[key].append(item.batch[key])
            for key in batch.non_tensor_batch.keys():
                expanded_non_tensors[key].append(item.non_tensor_batch[key])
            expanded_non_tensors["trajectory_group_id"].append(None)
            expanded_non_tensors["is_reward_source"].append(True)
        else:
            # Multi-trajectory output, expand each trajectory
            if isinstance(multi_traj, dict):
                multi_traj = MultiTrajectoryOutput.model_validate(multi_traj)

            for traj in multi_traj.trajectories:
                # Pad and add this trajectory
                traj_tensors = _pad_trajectory(
                    traj, tokenizer, prompt_length, response_length
                )
                for key in batch.batch.keys():
                    if key in traj_tensors:
                        expanded_tensors[key].append(traj_tensors[key])
                    else:
                        # Copy from original (e.g., multi_modal_inputs)
                        expanded_tensors[key].append(item.batch[key])

                for key in batch.non_tensor_batch.keys():
                    if key == "tool_extra_fields":
                        # Update extra fields for this trajectory
                        traj_extra = dict(extra)
                        traj_extra["trajectory_id"] = traj.trajectory_id
                        traj_extra["trajectory_role"] = traj.role
                        expanded_non_tensors[key].append(traj_extra)
                    else:
                        expanded_non_tensors[key].append(item.non_tensor_batch[key])

                expanded_non_tensors["trajectory_group_id"].append(
                    multi_traj.trajectory_group_id
                )
                expanded_non_tensors["is_reward_source"].append(traj.is_reward_source)

    # Rebuild the batch
    stacked_tensors = {
        key: torch.stack(vals, dim=0) for key, vals in expanded_tensors.items() if vals
    }
    stacked_non_tensors = {
        key: np.array(vals, dtype=object) for key, vals in expanded_non_tensors.items()
    }

    return DataProto(
        batch=TensorDict(
            stacked_tensors,
            batch_size=len(stacked_tensors[list(stacked_tensors.keys())[0]]),
        ),
        non_tensor_batch=stacked_non_tensors,
        meta_info=batch.meta_info,
    )


def _pad_trajectory(
    traj: TrajectoryOutput,
    tokenizer: Any,
    prompt_length: int,
    response_length: int,
) -> dict[str, torch.Tensor]:
    """Pad a trajectory's sequences to the required lengths."""
    tokenizer.padding_side = "left"
    prompt_output = tokenizer.pad(
        {"input_ids": traj.prompt_ids},
        padding="max_length",
        max_length=prompt_length,
        return_tensors="pt",
        return_attention_mask=True,
    )

    tokenizer.padding_side = "right"
    response_output = tokenizer.pad(
        {"input_ids": traj.response_ids},
        padding="max_length",
        max_length=response_length,
        return_tensors="pt",
        return_attention_mask=True,
    )

    response_mask_output = tokenizer.pad(
        {"input_ids": traj.response_mask},
        padding="max_length",
        max_length=response_length,
        return_tensors="pt",
        return_attention_mask=False,
    )

    prompt_ids = prompt_output["input_ids"].squeeze(0)
    response_ids = response_output["input_ids"].squeeze(0)
    response_mask = response_mask_output["input_ids"].squeeze(0) * response_output[
        "attention_mask"
    ].squeeze(0)
    attention_mask = torch.cat([
        prompt_output["attention_mask"].squeeze(0),
        response_output["attention_mask"].squeeze(0),
    ])
    input_ids = torch.cat([prompt_ids, response_ids])

    # Compute position IDs
    position_ids = torch.zeros_like(input_ids)
    valid_mask = attention_mask.bool()
    position_ids[valid_mask] = torch.arange(valid_mask.sum().item())

    return {
        "prompts": prompt_ids,
        "responses": response_ids,
        "response_mask": response_mask,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def postprocess_multi_trajectory_rewards(
    batch: DataProto,
    reward_key: str = "rm_scores",
) -> DataProto:
    """Post-process rewards for multi-trajectory batches.

    This function:
    1. Broadcasts rewards within trajectory groups
    2. Computes metrics about trajectory groups

    Call this after reward computation but before advantage computation.

    Args:
        batch: DataProto with rewards computed.
        reward_key: Key for reward tensor.

    Returns:
        DataProto: Updated batch with broadcasted rewards.
    """
    batch = broadcast_reward_by_trajectory_group(batch, reward_key=reward_key)
    return batch


def get_multi_trajectory_metrics(batch: DataProto) -> dict[str, float]:
    """Get metrics related to multi-trajectory training.

    Args:
        batch: DataProto with trajectory group information.

    Returns:
        dict: Metrics about trajectory groups.
    """
    return compute_trajectory_group_metrics(batch)


# Example integration with ray_trainer.py fit() loop:
#
# In fit(), after generating sequences and computing rewards:
#
# ```python
# from verl.experimental.multi_trajectory import (
#     expand_multi_trajectory_batch,
#     postprocess_multi_trajectory_rewards,
#     get_multi_trajectory_metrics,
# )
#
# # After rollout generation
# gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
#
# # Expand multi-trajectory outputs into separate rows
# gen_batch_output = expand_multi_trajectory_batch(
#     gen_batch_output,
#     self.tokenizer,
#     self.config.actor_rollout_ref.rollout.prompt_length,
#     self.config.actor_rollout_ref.rollout.response_length,
# )
#
# batch = batch.union(gen_batch_output)
#
# # ... compute rewards ...
#
# # Broadcast rewards within trajectory groups
# batch = postprocess_multi_trajectory_rewards(batch)
#
# # ... compute advantages (GRPO will naturally group by uid) ...
#
# # Log multi-trajectory metrics
# metrics.update(get_multi_trajectory_metrics(batch))
# ```
