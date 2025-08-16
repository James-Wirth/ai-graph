import logging
import networkx as nx
import datetime as dt
import uuid

from typing import Any, Dict, Optional, List, Tuple
from pydantic import BaseModel

from aigraph.core.agents import Agent, LLMAgent
from aigraph.core.tools import ToolRegistry, ToolResult

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
        self.logger = logging.getLogger("aigraph.runner")

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
                break

            agent: Agent = self.graph.nodes[current_node]["agent"]
            node_def = self.graph.nodes[current_node].get("node_def")
            tools_used: List[ToolResult] = []
            prompt_text: Optional[str] = None

            if node_def:
                specs = []
                for t in (node_def.tools or []):
                    if isinstance(t, dict):
                        specs.append(t)
                    elif isinstance(t, str):
                        specs.append({"name": t, "mode": "interactive"})
                ctx.variables.setdefault("tools", {})
                for spec in specs:
                    if spec.get("mode", "interactive") == "interactive":
                        continue
                    name = spec["name"]
                    alias = spec.get("as") or spec.get("alias") or name

                    def _get_path(root, path: str):
                        cur = root
                        for part in path.split("."):
                            if part == "":
                                continue
                            cur = (cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None))
                            if cur is None:
                                break
                        return cur

                    def _resolve(argmap, last_output, vars):
                        payload = getattr(last_output, "payload", last_output)
                        out = {}
                        for k, v in (argmap or {}).items():
                            if isinstance(v, str) and v.startswith("$"):
                                if v == "$input":
                                    out[k] = payload
                                elif v.startswith("$input."):
                                    out[k] = _get_path(payload, v[len("$input."):])
                                elif v == "$context":
                                    out[k] = vars
                                elif v.startswith("$context."):
                                    out[k] = _get_path(vars, v[len("$context."):])
                                else:
                                    out[k] = v
                            else:
                                out[k] = v
                        return out

                    inputs = _resolve(spec.get("argmap", {}), last_output, ctx.variables)
                    if isinstance(agent, LLMAgent) and agent.allowed_tools and name not in agent.allowed_tools:
                        raise PermissionError(
                            f"[{agent.name}] Pre-tool '{name}' not allowed. Allowed: {sorted(agent.allowed_tools)}"
                        )
                    tr = self.tool_registry.get(name).call(inputs)
                    ctx.variables["tools"][alias] = tr.model_dump()
                    if spec.get("required") and not tr.success:
                        raise RuntimeError(f"Required pre-tool '{name}' failed: {tr.error}")

            if isinstance(agent, LLMAgent):
                prompt_text = agent.build_prompt(last_output, ctx.variables)
                messages = agent.build_messages(last_output, ctx.variables)
                output = agent._llm_round(messages)
            else:
                output = agent.process(last_output, ctx.variables)

            if isinstance(agent, LLMAgent) and agent.max_tool_rounds > 0:
                rounds = 0
                per_tool_caps: Dict[str, int] = {}
                if node_def:
                    for t in (node_def.tools or []):
                        if isinstance(t, dict) and t.get("mode", "interactive") == "interactive" and "max_calls" in t:
                            per_tool_caps[t["name"]] = int(t["max_calls"])
                per_tool_counts: Dict[str, int] = {}

                while rounds < agent.max_tool_rounds and getattr(output, "tool_call", None):
                    try:
                        called = getattr(output.tool_call, "name", None)
                        if called in per_tool_caps:
                            c = per_tool_counts.get(called, 0)
                            if c >= per_tool_caps[called]:
                                break
                            per_tool_counts[called] = c + 1

                        tr = self._maybe_execute_tool(agent, output.tool_call)
                        tools_used.append(tr)
                        ctx.variables["__tool_last__"] = {
                            "name": tr.name,
                            "input": tr.input,
                            "output": tr.output,
                            "success": tr.success,
                            "error": tr.error,
                            "metadata": tr.metadata,
                        }

                        output = agent.process(last_output, ctx.variables)
                        prompt_text = agent.build_prompt(last_output, ctx.variables)
                        messages = agent.build_messages(last_output, ctx.variables)
                        output = agent._llm_round(messages)
                        rounds += 1

                        if not getattr(output, "tool_call", None):
                            break

                    except Exception as e:
                        ctx.variables["__tool_last__"] = {
                            "name": getattr(output.tool_call, "name", "?"),
                            "success": False,
                            "error": str(e),
                        }
                        output = agent.process(last_output, ctx.variables)
                        prompt_text = agent.build_prompt(last_output, ctx.variables)
                        messages = agent.build_messages(last_output, ctx.variables)
                        output = agent._llm_round(messages)
                        rounds += 1

                ctx.variables.pop("__tool_last__", None)

            neighbors = list(self.graph.neighbors(current_node))
            next_node, rationale, confidence = agent.route(neighbors, ctx.variables, output)

            ctx.record_step(
                node=current_node,
                prompt=prompt_text,
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

            is_router = isinstance(agent, LLMAgent) and bool(getattr(agent, "edges", {}))
            if not is_router:
                last_output = output

            step += 1

        return last_output, ctx
