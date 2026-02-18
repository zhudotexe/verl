"""
Tools for spawning and joining trajectories within an agent loop.

These tools allow an LLM agent to spawn parallel worker trajectories
and aggregate their results.
"""

import json
import logging
from typing import Any

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)


# Tool schemas for OpenAI function calling format
SPAWN_TRAJECTORY_SCHEMA = OpenAIFunctionToolSchema.model_validate({
    "type": "function",
    "function": {
        "name": "spawn_trajectory",
        "description": (
            "Spawn a new parallel worker trajectory to work on a subtask. "
            "The spawned trajectory runs asynchronously and can use tools. "
            "Use join_trajectories to retrieve results when ready. "
            "All spawned trajectories in a group share the same final reward."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task/prompt for the spawned worker trajectory.",
                },
                "agent_loop": {
                    "type": "string",
                    "description": "Agent loop type to use (default: tool_agent).",
                    "default": "tool_agent",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Optional system prompt for the worker.",
                },
            },
            "required": ["prompt"],
        },
    },
})

JOIN_TRAJECTORIES_SCHEMA = OpenAIFunctionToolSchema.model_validate({
    "type": "function",
    "function": {
        "name": "join_trajectories",
        "description": (
            "Wait for spawned trajectories to complete and retrieve their results. "
            "Pass the trajectory IDs returned by spawn_trajectory. "
            "Returns the final responses from each trajectory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "trajectory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of trajectory IDs to wait for.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Optional timeout in seconds.",
                },
            },
            "required": ["trajectory_ids"],
        },
    },
})


class SpawnTrajectoryTool(BaseTool):
    """Tool for spawning a new parallel worker trajectory.

    When called, this tool spawns a new trajectory that runs asynchronously.
    The spawned trajectory can use tools and generate its own responses.
    The result can be retrieved later using JoinTrajectoriesTool.
    """

    def __init__(
        self, config: dict = None, tool_schema: OpenAIFunctionToolSchema = None
    ):
        super().__init__(
            config=config or {"type": "native"},
            tool_schema=tool_schema or SPAWN_TRAJECTORY_SCHEMA,
        )

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        agent_data: Any = None,
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        """Spawn a new trajectory.

        Args:
            instance_id: Tool instance ID.
            parameters: Parameters with 'prompt', optional 'agent_loop', 'system_prompt'.
            agent_data: AgentData containing the trajectory_manager.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (ToolResponse with trajectory_id, reward=0.0, metrics).
        """
        # Get trajectory manager from agent_data
        trajectory_manager = getattr(agent_data, "trajectory_manager", None)
        if trajectory_manager is None:
            # Check extra_fields
            trajectory_manager = (
                agent_data.extra_fields.get("trajectory_manager")
                if agent_data
                else None
            )

        if trajectory_manager is None:
            return (
                ToolResponse(
                    text=(
                        "Error: TrajectoryManager not available. Cannot spawn"
                        " trajectories."
                    )
                ),
                0.0,
                {},
            )

        prompt = parameters.get("prompt", "")
        agent_loop = parameters.get("agent_loop", "tool_agent")
        system_prompt = parameters.get("system_prompt")

        try:
            handle = await trajectory_manager.spawn(
                prompt=prompt,
                agent_loop=agent_loop,
                system_prompt=system_prompt,
                tools_kwargs=agent_data.tools_kwargs if agent_data else {},
            )

            result = {
                "trajectory_id": handle.trajectory_id,
                "status": handle.status,
                "message": (
                    f"Spawned trajectory {handle.trajectory_id} with"
                    f" agent_loop={agent_loop}"
                ),
            }

            return (
                ToolResponse(text=json.dumps(result)),
                0.0,
                {"spawned_trajectory_id": handle.trajectory_id},
            )

        except Exception as e:
            logger.error(f"Failed to spawn trajectory: {e}")
            return ToolResponse(text=f"Error spawning trajectory: {e}"), 0.0, {}


class JoinTrajectoriesTool(BaseTool):
    """Tool for waiting on and retrieving results from spawned trajectories.

    This tool blocks until the specified trajectories complete, then
    returns their final responses.
    """

    def __init__(
        self, config: dict = None, tool_schema: OpenAIFunctionToolSchema = None
    ):
        super().__init__(
            config=config or {"type": "native"},
            tool_schema=tool_schema or JOIN_TRAJECTORIES_SCHEMA,
        )

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        agent_data: Any = None,
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        """Join spawned trajectories and get their results.

        Args:
            instance_id: Tool instance ID.
            parameters: Parameters with 'trajectory_ids', optional 'timeout'.
            agent_data: AgentData containing the trajectory_manager.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (ToolResponse with results, reward=0.0, metrics).
        """
        trajectory_manager = getattr(agent_data, "trajectory_manager", None)
        if trajectory_manager is None:
            trajectory_manager = (
                agent_data.extra_fields.get("trajectory_manager")
                if agent_data
                else None
            )

        if trajectory_manager is None:
            return (
                ToolResponse(
                    text=(
                        "Error: TrajectoryManager not available. Cannot join"
                        " trajectories."
                    )
                ),
                0.0,
                {},
            )

        trajectory_ids = parameters.get("trajectory_ids", [])
        timeout = parameters.get("timeout")

        if not trajectory_ids:
            return ToolResponse(text="Error: No trajectory_ids provided."), 0.0, {}

        try:
            results = await trajectory_manager.join(trajectory_ids, timeout=timeout)

            # Format results for the LLM
            formatted_results = []
            for i, (tid, result) in enumerate(zip(trajectory_ids, results)):
                # Decode the response to text
                response_text = (
                    kwargs.get("tokenizer", trajectory_manager.tokenizer).decode(
                        result.response_ids, skip_special_tokens=True
                    )
                    if hasattr(result, "response_ids")
                    else str(result)
                )

                formatted_results.append({
                    "trajectory_id": tid,
                    "response": response_text,
                    "role": result.role,
                })

            return (
                ToolResponse(text=json.dumps(formatted_results, indent=2)),
                0.0,
                {"joined_count": len(results)},
            )

        except asyncio.TimeoutError:
            return (
                ToolResponse(
                    text=f"Error: Timeout waiting for trajectories: {trajectory_ids}"
                ),
                0.0,
                {},
            )
        except Exception as e:
            logger.error(f"Failed to join trajectories: {e}")
            return ToolResponse(text=f"Error joining trajectories: {e}"), 0.0, {}


# Import asyncio for the timeout error
import asyncio
