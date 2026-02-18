"""
Multi-trajectory agent loop that supports spawning child trajectories.

This agent loop extends ToolAgentLoop to allow the LLM to spawn parallel
worker trajectories using the spawn_trajectory and join_trajectories tools.
"""

import logging
from typing import Any, Optional
from uuid import uuid4

from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
    register,
)
from verl.experimental.agent_loop.tool_agent_loop import AgentData, ToolAgentLoop
from verl.utils.rollout_trace import rollout_trace_op
from .output import TrajectoryOutput
from .tools import JoinTrajectoriesTool, SpawnTrajectoryTool
from .trajectory_manager import TrajectoryManager

logger = logging.getLogger(__name__)


@register("multi_trajectory_agent")
class MultiTrajectoryAgentLoop(ToolAgentLoop):
    """Agent loop that supports spawning multiple parallel trajectories.

    This extends ToolAgentLoop to add:
    - spawn_trajectory tool: Spawn a child trajectory to work on a subtask
    - join_trajectories tool: Wait for and collect results from child trajectories
    - TrajectoryManager: Manages the lifecycle of spawned trajectories

    All trajectories in a group (parent + children) share the same reward
    based on the final answer.
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)

        # Add multi-trajectory tools
        self.spawn_tool = SpawnTrajectoryTool()
        self.join_tool = JoinTrajectoriesTool()

        # Add to tools dict
        self.tools["spawn_trajectory"] = self.spawn_tool
        self.tools["join_trajectories"] = self.join_tool

        # Add tool schemas
        self.tool_schemas.append(
            self.spawn_tool.tool_schema.model_dump(
                exclude_unset=True, exclude_none=True
            )
        )
        self.tool_schemas.append(
            self.join_tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
        )

        # Multi-trajectory config
        config = trainer_config.config
        multi_traj_config = config.actor_rollout_ref.rollout.get("multi_trajectory", {})
        self.max_concurrent_trajectories = multi_traj_config.get("max_concurrent", 8)
        self.enable_recursive_spawn = multi_traj_config.get("recursive", True)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run the agent loop with multi-trajectory support.

        This method sets up the TrajectoryManager and runs the standard
        tool agent loop. Child trajectories can be spawned via tool calls.

        Args:
            sampling_params: LLM sampling parameters.
            **kwargs: Dataset fields including raw_prompt, tools_kwargs, etc.

        Returns:
            AgentLoopOutput: The agent's output, potentially including
                            information about spawned trajectories.
        """
        # Generate unique IDs for this trajectory group
        trajectory_group_id = kwargs.get("trajectory_group_id", uuid4().hex)
        parent_trajectory_id = f"{trajectory_group_id}_parent"

        # Create trajectory manager
        trajectory_manager = TrajectoryManager(
            trajectory_group_id=trajectory_group_id,
            parent_trajectory_id=parent_trajectory_id,
            agent_loop_runner=self._run_child_trajectory,
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config,
            max_concurrent_trajectories=self.max_concurrent_trajectories,
        )

        # Store in kwargs for tool access
        kwargs["trajectory_manager"] = trajectory_manager
        kwargs["trajectory_group_id"] = trajectory_group_id

        # Run the standard tool agent loop
        output = await super().run(sampling_params, **kwargs)

        # Check if any trajectories were spawned
        spawned_handles = trajectory_manager.get_all_handles()
        if spawned_handles:
            # Build multi-trajectory output
            parent_traj = TrajectoryOutput(
                prompt_ids=output.prompt_ids,
                response_ids=output.response_ids,
                response_mask=output.response_mask,
                response_logprobs=output.response_logprobs,
                trajectory_id=parent_trajectory_id,
                role="aggregator",
                is_reward_source=True,
                extra_fields=output.extra_fields,
            )

            multi_output = trajectory_manager.build_multi_trajectory_output(
                parent_output=parent_traj,
                reward_source="parent",
            )

            # Store multi-trajectory info in extra_fields
            output.extra_fields["multi_trajectory_output"] = multi_output.model_dump()
            output.extra_fields["trajectory_group_id"] = trajectory_group_id
            output.extra_fields["is_reward_source"] = True
            output.extra_fields["spawned_trajectory_ids"] = [
                h.trajectory_id for h in spawned_handles
            ]

        return output

    async def _run_child_trajectory(
        self,
        raw_prompt: list[dict[str, Any]],
        agent_name: str = "tool_agent",
        tools_kwargs: dict[str, Any] = None,
        extra_info: dict[str, Any] = None,
        trajectory_manager: Optional[TrajectoryManager] = None,
        **kwargs,
    ) -> AgentLoopOutput:
        """Run a child trajectory.

        This is called by the TrajectoryManager when spawning child trajectories.

        Args:
            raw_prompt: Chat messages for the child trajectory.
            agent_name: Agent loop to use for the child.
            tools_kwargs: Tool kwargs.
            extra_info: Extra context.
            trajectory_manager: Optional manager for nested spawning.
            **kwargs: Additional arguments.

        Returns:
            AgentLoopOutput: The child trajectory's output.
        """
        # Default sampling params for child trajectories
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            logprobs=config.calculate_log_probs,
        )

        # Prepare kwargs for the child
        child_kwargs = {
            "raw_prompt": raw_prompt,
            "tools_kwargs": tools_kwargs or {},
            "extra_info": extra_info or {},
        }

        # If recursive spawning is enabled, allow children to spawn
        if self.enable_recursive_spawn and trajectory_manager is not None:
            child_kwargs["trajectory_manager"] = trajectory_manager

        # Run with the tool agent loop (not multi-trajectory to avoid infinite recursion)
        # Use parent class's run method
        return await ToolAgentLoop.run(self, sampling_params, **child_kwargs)


class MultiTrajectoryAgentData(AgentData):
    """Extended AgentData that includes trajectory management information."""

    def __init__(
        self, *args, trajectory_manager: Optional[TrajectoryManager] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.trajectory_manager = trajectory_manager
