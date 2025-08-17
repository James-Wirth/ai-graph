# AIGraph


**AIGraph** is a lighweight, decorator-driven framework for building complex, graph-based agentic workflows. 

It combines the ergonomics of writing plain Python functions with the robustness of typed, inspectable graphs, letting you design and run agentic systems with minimal ceremony.

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

### 1. Payload Flow

Every run carries a payload (an arbitrary Python object or Pydantic model) through the graph. 

- `ag.data()` fetches the current payload
- `ag.set_data(value)` updates the payload for downstream nodes

Example:

```python
@ag.start
def begin():
    text = ag.data()
    ag.set_data({"text": text, "lang": "en"})
```

### 2. Execution Context (`ctx`)

Each run carries a context dictionary, available via `ag.vars()`. This acts as a shared scratch space for the workflow, whose values persist across nodes. 

Example:

`ag.vars()["topic_hint"] = "general relativity"`

### 3. Logging with artifacts (via `emit()`)

Nodes can produce artifacts: structured events that are appended to the run log. 

Example:

```python
ag.emit({"kind":"metrics", "token_usage": 42})
```

### 4. Execution History

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



