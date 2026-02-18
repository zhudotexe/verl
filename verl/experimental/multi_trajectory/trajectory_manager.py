"""
TrajectoryManager handles spawning and joining of child trajectories.

This manager is instantiated per-request and allows an agent to spawn
parallel worker trajectories, wait for their completion, and aggregate results.
"""

import asyncio
import logging
from typing import Any, Optional
from uuid import uuid4

from .output import MultiTrajectoryOutput, SpawnedTrajectoryHandle, TrajectoryOutput

logger = logging.getLogger(__name__)


class TrajectoryManager:
    """Manages spawned trajectories within a single agent loop execution.

    The TrajectoryManager is created for each top-level trajectory and allows
    that trajectory to spawn child trajectories that run in parallel. All
    trajectories in the same group share the same reward.

    Usage:
        manager = TrajectoryManager(trajectory_group_id, agent_loop_runner)

        # Spawn child trajectories
        handle1 = await manager.spawn(prompt="Analyze approach A", agent_loop="tool_agent")
        handle2 = await manager.spawn(prompt="Analyze approach B", agent_loop="tool_agent")

        # Continue with parent work...

        # Join to get results
        results = await manager.join([handle1.trajectory_id, handle2.trajectory_id])
    """

    def __init__(
        self,
        trajectory_group_id: str,
        parent_trajectory_id: str,
        agent_loop_runner: Any,  # Callable to run an agent loop
        server_manager: Any,
        tokenizer: Any,
        processor: Any,
        config: Any,
        max_concurrent_trajectories: int = 16,
    ):
        """Initialize the TrajectoryManager.

        Args:
            trajectory_group_id: Unique ID for this group of related trajectories.
            parent_trajectory_id: ID of the parent trajectory that owns this manager.
            agent_loop_runner: Async callable to run an agent loop given a prompt.
            server_manager: LLM server manager for generation.
            tokenizer: Tokenizer for the model.
            processor: Processor for multimodal data.
            config: Configuration object.
            max_concurrent_trajectories: Maximum number of concurrent spawned trajectories.
        """
        self.trajectory_group_id = trajectory_group_id
        self.parent_trajectory_id = parent_trajectory_id
        self.agent_loop_runner = agent_loop_runner
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.max_concurrent_trajectories = max_concurrent_trajectories

        # Track spawned trajectories
        self._handles: dict[str, SpawnedTrajectoryHandle] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._results: dict[str, TrajectoryOutput] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_trajectories)

    async def spawn(
        self,
        prompt: str | list[dict[str, Any]],
        agent_loop: str = "tool_agent",
        system_prompt: Optional[str] = None,
        tools_kwargs: Optional[dict[str, Any]] = None,
        extra_info: Optional[dict[str, Any]] = None,
    ) -> SpawnedTrajectoryHandle:
        """Spawn a new child trajectory.

        The spawned trajectory runs asynchronously and can be joined later
        to retrieve its results.

        Args:
            prompt: The prompt for the child trajectory. Can be a string or
                   a list of chat messages.
            agent_loop: The agent loop type to use (e.g., "tool_agent").
            system_prompt: Optional system prompt override.
            tools_kwargs: Additional kwargs passed to tool calls.
            extra_info: Additional context info.

        Returns:
            SpawnedTrajectoryHandle: A handle to track and join the trajectory.
        """
        trajectory_id = f"{self.trajectory_group_id}_{uuid4().hex[:8]}"

        handle = SpawnedTrajectoryHandle(
            trajectory_id=trajectory_id,
            trajectory_group_id=self.trajectory_group_id,
            agent_loop=agent_loop,
            status="pending",
        )
        self._handles[trajectory_id] = handle

        # Build the raw_prompt
        if isinstance(prompt, str):
            raw_prompt = []
            if system_prompt:
                raw_prompt.append({"role": "system", "content": system_prompt})
            raw_prompt.append({"role": "user", "content": prompt})
        else:
            raw_prompt = prompt

        # Create the async task
        task = asyncio.create_task(
            self._run_trajectory(
                trajectory_id=trajectory_id,
                raw_prompt=raw_prompt,
                agent_loop=agent_loop,
                tools_kwargs=tools_kwargs or {},
                extra_info=extra_info or {},
            )
        )
        self._tasks[trajectory_id] = task
        handle.status = "running"

        return handle

    async def _run_trajectory(
        self,
        trajectory_id: str,
        raw_prompt: list[dict[str, Any]],
        agent_loop: str,
        tools_kwargs: dict[str, Any],
        extra_info: dict[str, Any],
    ) -> TrajectoryOutput:
        """Run a spawned trajectory.

        Args:
            trajectory_id: Unique ID for this trajectory.
            raw_prompt: The chat messages for the trajectory.
            agent_loop: Agent loop type.
            tools_kwargs: Tool kwargs.
            extra_info: Extra context info.

        Returns:
            TrajectoryOutput: The trajectory's output.
        """
        async with self._semaphore:
            try:
                # Run the agent loop
                output = await self.agent_loop_runner(
                    raw_prompt=raw_prompt,
                    agent_name=agent_loop,
                    tools_kwargs=tools_kwargs,
                    extra_info=extra_info,
                    trajectory_manager=None,  # Child trajectories can spawn their own
                )

                # Convert to TrajectoryOutput
                result = TrajectoryOutput(
                    prompt_ids=output.prompt_ids,
                    response_ids=output.response_ids,
                    response_mask=output.response_mask,
                    response_logprobs=output.response_logprobs,
                    trajectory_id=trajectory_id,
                    role="worker",
                    is_reward_source=False,
                    extra_fields=output.extra_fields,
                )

                self._results[trajectory_id] = result
                self._handles[trajectory_id].status = "completed"
                self._handles[trajectory_id].result = result
                return result

            except Exception as e:
                logger.error(f"Trajectory {trajectory_id} failed: {e}")
                self._handles[trajectory_id].status = "failed"
                raise

    async def join(
        self,
        trajectory_ids: list[str],
        timeout: Optional[float] = None,
    ) -> list[TrajectoryOutput]:
        """Wait for spawned trajectories to complete and return their results.

        Args:
            trajectory_ids: List of trajectory IDs to wait for.
            timeout: Optional timeout in seconds.

        Returns:
            List[TrajectoryOutput]: Results from each trajectory, in order.

        Raises:
            asyncio.TimeoutError: If timeout is exceeded.
            ValueError: If a trajectory ID is not found.
        """
        tasks = []
        for tid in trajectory_ids:
            if tid not in self._tasks:
                raise ValueError(f"Unknown trajectory ID: {tid}")
            tasks.append(self._tasks[tid])

        if timeout:
            done, pending = await asyncio.wait(tasks, timeout=timeout)
            if pending:
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                raise asyncio.TimeoutError(f"Timed out waiting for trajectories")
        else:
            await asyncio.gather(*tasks)

        return [self._results[tid] for tid in trajectory_ids]

    async def join_all(self, timeout: Optional[float] = None) -> list[TrajectoryOutput]:
        """Wait for all spawned trajectories to complete.

        Args:
            timeout: Optional timeout in seconds.

        Returns:
            List[TrajectoryOutput]: Results from all trajectories.
        """
        return await self.join(list(self._handles.keys()), timeout=timeout)

    def get_handle(self, trajectory_id: str) -> Optional[SpawnedTrajectoryHandle]:
        """Get the handle for a spawned trajectory."""
        return self._handles.get(trajectory_id)

    def get_all_handles(self) -> list[SpawnedTrajectoryHandle]:
        """Get handles for all spawned trajectories."""
        return list(self._handles.values())

    def get_completed_results(self) -> list[TrajectoryOutput]:
        """Get results from all completed trajectories."""
        return list(self._results.values())

    def build_multi_trajectory_output(
        self,
        parent_output: TrajectoryOutput,
        reward_source: str = "parent",
    ) -> MultiTrajectoryOutput:
        """Build a MultiTrajectoryOutput combining parent and child trajectories.

        Args:
            parent_output: The output from the parent (aggregator) trajectory.
            reward_source: Which trajectory determines reward: "parent", "last", or a trajectory_id.

        Returns:
            MultiTrajectoryOutput: Combined output with all trajectories.
        """
        # Collect all trajectory outputs
        trajectories = list(self._results.values())

        # Add parent as the aggregator
        parent_output.role = "aggregator"
        trajectories.append(parent_output)

        # Determine reward source
        if reward_source == "parent":
            reward_source_index = len(trajectories) - 1
        elif reward_source == "last":
            reward_source_index = len(trajectories) - 1
        else:
            # Find by trajectory_id
            reward_source_index = -1
            for i, traj in enumerate(trajectories):
                if traj.trajectory_id == reward_source:
                    reward_source_index = i
                    break
            if reward_source_index < 0:
                reward_source_index = len(trajectories) - 1

        output = MultiTrajectoryOutput(
            trajectories=trajectories,
            trajectory_group_id=self.trajectory_group_id,
            reward_source_index=reward_source_index,
        )
        output.mark_reward_source()

        return output
