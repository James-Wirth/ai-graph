import logging
import networkx as nx
import datetime as dt
import uuid

from typing import Any, Dict, Optional, List, Tuple
from pydantic import BaseModel

from aigraph.agents import Agent, LLMAgent
from aigraph.tools import ToolRegistry, ToolResult

class ExecutionContext:
    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or str(uuid.uuid4())
        self.history: List[Dict[str, Any]] = []
        self.variables: Dict[str, Any] = {}
        self.created_at = dt.datetime.now(dt.UTC)

    def record_step(
        self,
        node: Any,
        prompt: Optional[str],
        result: BaseModel,
        neighbours: List[Any],
        decision: Dict[str, Any],
        tools_used: Optional[List[ToolResult]] = None,
    ):
        self.history.append({
            'timestamp': dt.datetime.now(dt.UTC).isoformat(),
            'node': node,
            'prompt': prompt,
            'result': result.model_dump() if hasattr(result, 'model_dump') else getattr(result, '__dict__', None),
            'neighbours': neighbours,
            'decision': decision,
            'tools_used': [t.model_dump() for t in (tools_used or [])],
        })

class GraphRunner:
    def __init__(
        self,
        graph: nx.DiGraph,
        tool_registry: ToolRegistry = None,
        max_steps: int = 10,
    ):
        self.graph = graph
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_steps = max_steps
        self.logger = logging.getLogger("llmgraph.runner")

        for n, data in self.graph.nodes(data=True):
            agent = data.get("agent")
            if not isinstance(agent, Agent):
                raise AssertionError(f"Node '{n}' does not contain a valid Agent (found: {type(agent)})")

    def _maybe_execute_tool(self, agent: LLMAgent, tool_call_obj: Any) -> ToolResult:
        tool_name = getattr(tool_call_obj, "name", None)
        tool_input = getattr(tool_call_obj, "input", {}) or {}
        if tool_name not in agent.allowed_tools:
            raise PermissionError(
                f"[{agent.name}] Tool '{tool_name}' not allowed. Allowed: {sorted(agent.allowed_tools)}"
            )
        tool = self.tool_registry.get(tool_name)
        return tool.call(tool_input)

    def run(self, input_model: BaseModel, start_node: Any) -> Tuple[BaseModel, ExecutionContext]:
        if start_node not in self.graph:
            raise AssertionError(f"Start node '{start_node}' not in graph")
        ctx = ExecutionContext()
        current_node = start_node
        last_output: Optional[BaseModel] = input_model
        step = 0

        self.logger.info("Starting workflow run %s at node %s", ctx.run_id, current_node)

        while True:
            if step >= self.max_steps:
                self.logger.warning("Max steps %d reached, stopping.", self.max_steps)
                break

            agent: Agent = self.graph.nodes[current_node]["agent"]
            tools_used: List[ToolResult] = []
            output = agent.process(last_output, ctx.variables)

            if isinstance(agent, LLMAgent) and agent.max_tool_rounds > 0:
                rounds = 0
                while rounds < agent.max_tool_rounds and getattr(output, "tool_call", None):
                    try:
                        tr = self._maybe_execute_tool(agent, output.tool_call)
                        tools_used.append(tr)
                        ctx.variables.setdefault("tools", {})[tr.name] = tr.model_dump()
                        output = agent.process(last_output, ctx.variables)
                        rounds += 1
                    except Exception as e:
                        self.logger.error("Tool execution failed for agent '%s': %s", agent.name, e)
                        ctx.variables.setdefault("tool_errors", []).append(str(e))
                        output = agent.process(last_output, ctx.variables)
                        rounds += 1

            neighbors = list(self.graph.neighbors(current_node))
            next_node, rationale, confidence = agent.route(neighbors, ctx.variables, output)

            ctx.record_step(
                node=current_node,
                prompt=None,
                result=output,
                neighbours=neighbors,
                decision={"next_node": next_node, "rationale": rationale, "confidence": confidence},
                tools_used=tools_used,
            )

            if next_node is None:
                self.logger.info("Terminal at node %s (%s)", current_node, rationale)
                last_output = output
                break

            if next_node not in neighbors:
                raise AssertionError(f"Agent '{agent.name}' chose invalid neighbor '{next_node}' from {neighbors}")

            current_node = next_node
            last_output = output
            step += 1

        return last_output, ctx
