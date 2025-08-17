# AIGraph


**AIGraph** is a lighweight, decorator-driven framework for building complex, graph-based agentic workflows. 

It combines the ergonomics of writing plain Python functions with the robustness of typed, inspectable graphs, letting you design and run agentic systems with minimal ceremony.

## Example: Physics lesson with Newton and Einstein 

Here's an example of a simple graph-based workflow, which routes your Physics question to Newton or Einstein depending on the content. We'll show how to hook up some simple external tools.

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

```python
class Input(BaseModel):
    question: str
    data: Optional[Dict[str, float]] = None

@ag.start(next="routing_node")
def start_node(
    question: str = ag.From("question"),
    data: Optional[Dict[str, float]] = ag.From("data"),
) -> Input:
    """
    - Python code in the function body executes when the node is reached.
    - The function return is passed to the next node as the input.

    - In this example, we'll just format the input arguments into
      a nice Pydantic schema.
    """
    return Input(question=question, data=data)
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
    ${question}
    """,
)
def routing_node(
    question: str = ag.From("question"),
    data: Optional[Dict[str, float]] = ag.From("data"),
) -> Literal["newton_node", "einstein_node"]:
    """
    - return ag.accept_llm() simply passes on the structured LLM
      output to the next node, without any manual processing.
    """
    return ag.accept_llm()
```

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
    Question: ${question}
    """,
)
def newton_node(
    question: str = From("question"),
    data: Optional[Dict[str, float]] = ag.From("data"),
) -> Answer:
    return ag.accept_llm()


@ag.step(
    next="end_node",
    prompt="""
    You are Albert Einstein. Answer this question:
    Question: ${question}
    """,
)
def einstein_node(
    question: str = From("question"),
    data: Optional[Dict[str, float]] = ag.From("data"),
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
        "alias": "force_tool",
        "required": True,
        "argmap": {
            "mass": "$param.data.m",
            "acceleration": "$param.data.a",
        },
    }],
)
def newton_node(
    question: str = From("question"),
    data: Optional[Dict[str, float]] = From("data"),
    force: Optional[float] = ToolValue("force_tool", field="output"),
) -> Answer:
    return ag.accept_llm()
```


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

## Highlights

### Declarative Graph Construction

Define workflow nodes with decorators (`@start`, `@step`, `@route`, `@end`). Each node can prompt an LLM, run custom Python, and more!

### Automatic Graph Compilation

The API automatically compiles your code into a `networkx.DiGraph`. This makes it easy to view the graph structure and debug the workflow.

### Structured Inputs & Outputs

Robust schema validation using **Pydantic** models.

### Custom Tooling

Define tools that run automatically upon reaching a node, whose results are stored in a global `$context.tools` and can be dynamically injected into the LLM prompt. 

### Branching & Routing

`@route` nodes use LLMs (or custom Python selectors) to choose the next edge. The options are strongly typed.

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



