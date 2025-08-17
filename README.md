# AIGraph


**AIGraph** is a lighweight, decorator-driven framework for building complex, graph-based agentic workflows. 

AIGraph lets you write nodes as plain Python functions, but wires them up into a graph with typed inputs and outputs. Under the hood it uses Pydantic for validation, so every edge between nodes is schema-checked. You get the convenience of just writing functions, plus the safety of strong types, without having to deal with excessive boilerplate.

## Example: Physics lesson with Newton and Einstein 

Here's an example of a simple graph-based workflow, which routes your Physics question to Newton or Einstein depending on the content. We'll also show how to hook up some simple external tools.

<p align="center" style="margin: 30px 0;">
  <img src="https://github.com/user-attachments/assets/5ab111c8-771c-4c59-a7a2-70ce32a0bd94" alt="physics_workflow" width="500" />
</p>


### 1. Create the app

```python
import aigraph as ag

app = ag.App(
    name="Physics Lesson",
    model="llama3",
    temperature=0.1
)
```

### 2. Make a "start" node 

You can do whatever pre-processing you want in the entry point. In this example, we'll simply validate that the user input conforms to the BundledInput schema and pass it on to the next node.

```python
class ValidInput(BaseModel):
    question: str
    data: Optional[Dict[str, float]] = None

@ag.start(next="routing_node")
def start_node(payload: ValidInput = ag.Payload) -> ValidInput:
    return payload
```

We've specified the next node (`routing_node`) in the decorator. The compiler will automatically create the edge when the graph is run.

### 3. Choosing the best tutor with a "route" node

```python
@ag.route(
    prompt="""
    Classify the question and pick the best tutor:

    - "newton_node" for classical mechanics.
    - "einstein_node" for relativity.

    Question:
    ${param.question}
    """,
)
def routing_node(
    question: str                    = ag.FromPayload("question"),  # access payload entries directly 
    data: Optional[Dict[str, float]] = ag.FromPayload("data"),      # (e.g. for prompt injection)
) -> Literal["newton_node", "einstein_node"]:
    """
    - return ag.accept_llm() simply passes on the structured LLM
      output to the next node, without any manual processing.
    """
    return ag.accept_llm()
```

We've injected the question argument into the prompt with `${param.question}`. 
The routing agent returns either `"netwon_node"` or `"einstein_node"` depending on the question, and proceeds to that node. 

### 4.1 Implementing the answers with "step" nodes for Newton and Einstein.

```python
class Answer(BaseModel):
    answer: float
    explanation: str = Field(min_length=40, max_length=800)

@ag.step(
    next="end_node",
    prompt="""
    You are Isaac Newton. Answer this question:
    Question: ${param.question}
    """,
)
def newton_node(
    question: str                    = ag.FromPayload("question"),
    data: Optional[Dict[str, float]] = ag.FromPayload("data"),
) -> Answer:
    return ag.accept_llm()


@ag.step(
    next="end_node",
    prompt="""
    You are Albert Einstein. Answer this question:
    Question: ${param.question}
    """,
)
def einstein_node(
    question: str                    = ag.FromPayload("question"),
    data: Optional[Dict[str, float]] = ag.FromPayload("data"),
) -> Answer:
    return ag.accept_llm()
```

These steps implement the Newton and Einstein agents, which read the question in the payload and return a structured `Answer` output that is sent to the `finish_node`.

### 4.2 Adding tooling

You can augment nodes with custom tools. The results of the tools can be injected into the prompts. For example, we could give the Newton agent a tool that calculates force:

```python
def compute_force(mass: float, acceleration: float) -> Optional[float]:
    try:
        return float(mass) * float(acceleration)
    except Exception:
        return None

@ag.step(
    next="end_node",
    prompt="""
    You are Isaac Newton. Answer this question:

    Question: ${param.question}
    Force (tool result): ${tool.force} 
    """,
    tools=[{
        "name": "compute_force",
        "alias": "force",
        "required": True,
        "argmap": {
            "mass": "$param.data.m",
            "acceleration": "$param.data.a",
        },
    }],
)
def newton_node(
    question: str                    = ag.FromPayload("question"),
    data: Optional[Dict[str, float]] = ag.FromPayload("data"),
    force: Optional[float]           = ag.FromTool("force", field="output"),  # access tool outputs
) -> Answer:
    return ag.accept_llm()
```

The tool result has been injected into the prompt with `${tool.force}`.

### 5. Make an "end" node

```python
class Final(BaseModel):
    final: Answer

@ag.end
def end_node(payload: Answer = ag.Payload) -> Final:
    return Final(final=payload)
```
### 6. Running the graph

```python
run_spec = {
    "question": "A 5 kg box accelerates at 3 m/s^2. What is the net force?",
    "data": {"m": 5.0, "a": 3.0},
}
out, ctx = app.run(run_spec)
```

The output from the graph above:
```
"final": {
    "answer": 15.0,
    "explanation": "F = ma, where F is the net force, m is mass (5 kg), and a is acceleration (3 m/s^2). Plugging in the values, we get F = 5 kg * 3 m/s^2 = 15 N."
}
```

## Highlights

- **Declarative Graph Construction**: Define workflow nodes with decorators (`@start`, `@step`, `@route`, `@end`). Each node can prompt an LLM, run custom Python, and more!

- **Automatic Graph Compilation**: The API automatically compiles your code into a `networkx.DiGraph`. This makes it easy to view the graph structure and debug the workflow.

- **Structured Inputs & Outputs**: Robust schema validation using **Pydantic** models.

- **Custom Tooling**: Define tools that run automatically upon reaching a node, whose results are stored in a global `$context.tools` and can be dynamically injected into the LLM prompt. 

- **Branching & Routing**: `@route` nodes use LLMs (or custom Python selectors) to choose the next edge. The options are strongly typed.

## Execution Context & Artifacts 

### 1. Execution Context (`ctx`)

Each run has a shared context dictionary that persists across nodes. Use if for scratch data, etc.

Example:

```python
@ag.step
def stash_hint():
    ag.vars()["hint"] = "something"

@ag.step
def read_hint():
    hint = ag.vars().get("hint")
```

### 2. Logging with artifacts (via `emit()`)

Nodes can produce artifacts: structured events that are appended to the run log. 

Example:

```python
ag.emit({"kind":"metrics", "token_usage": 42})
```

### 3. Execution History

When a run finishes, `app.run(...)` returns both the final payload and the execution context (containing history and artifacts).

Example:

```python
output, ctx = app.run("Hi!")

print(ctx.history)
print(ctx.variables)
print(ctx.artifacts)
```

## Visualization 

It's easy to generate Graphviz visualizations of your agentic workflow.

```python
app.viz(history=ctx.history).save("workflow.png")
```



