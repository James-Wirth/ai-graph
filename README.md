# aigraph

[![status: active development](https://img.shields.io/badge/status-active--development-orange)]()

**aigraph** is a fairly lightweight, decorator-driven framework for building complex, graph-based agentic workflows. 

The idea is to let you write nodes as decorated Python functions. The compiler automatically wires them up into a graph with typed inputs and outputs, using Pydantic for schema validation. You get the convenience of just writing decorated functions, plus the safety of strong types, without having to deal with excessive boilerplate.

ai-graph supports a few key types of nodes: start nodes, end nodes, routing nodes (for decisions) and step nodes. 

## Usage

This library is under active development. 
Expect frequent changes and breaking updates!

```
git clone https://github.com/James-Wirth/ai-graph.git
cd aigraph
pip install -e ".[dev]"  
```

## Example: Physics lesson with Newton and Einstein 

Here's a (contrived) example of a simple graph-based workflow, which routes your Physics question to Newton or Einstein depending on the content. We'll also show how to hook up some simple external tools.

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

When the app is run with `app.run(...)`, the entry point of the graph is the function decorated with `@ag.start`.
In this minimal example, the start node will simply return the payload without any pre-processing.

We've specified the next node (`next="route"`) in the decorator. 
The compiler will automatically create the edge when the graph is run, sending the output of this function to the next node.

```python
class Input(BaseModel):
    question: str
    data: Optional[Dict[str, float]] = None

@ag.start(next="route")
def start(payload: Input) -> Input:
    # You could do some pre-processing here if you felt like it...
    return payload
```

### 3. Choosing the best tutor with a "route" node

Route nodes allow the LLM to make a choice on how to proceed through the graph. 
You can also manually override this in the function body (for debugging, simple cases, etc.)

In the example below, we've injected the question into the prompt with `{payload.question}`. 
The routing agent returns either `"newton"` or `"einstein"` depending on the question, and proceeds to that node. 

```python
@ag.route(cases=["newton", "einstein"]) 
def route(payload: Input) -> ag.Route[Literal["newton", "einstein"]]:

    # We'll let the LLM decide the route...
    # depending on the question subject (accessed via payload.question)
    return ag.Route.delegate(
        prompt=f"""
        Classify the question:
        - "newton" for classical mechanics
        - "einstein" for relativity

        Question: {payload.question}
        """
    )
```

### 4.1 Implementing the answers with "step" nodes for Newton and Einstein.

Step nodes (decorated with `@ag.step`) are the core building blocks of the graph.
In the example below, we've created two step nodes `newton` and `einstein`, which process the question and data (contained in the payload).

We'll define a Pydantic model `Answer` to which the LLM output will be constrained.
For brevity, both Newton and Einstein use the same output schema below (but this doesn't have to be the case).

```python
class Answer(BaseModel):
    answer: float
    explanation: str = Field(min_length=40, max_length=800)

@ag.step(next="end")
def newton(payload: Input) -> Answer:
    # Return the structured output (w/ schema "Answer") from the LLM
    return ag.llm(
        model=Answer,
        prompt=f"""
        You are Isaac Newton. 
        Answer this Physics question in your personal style:

        Question: {payload.question}
        Data: {payload.data}
        """
    )

@ag.step(next="end")
def einstein(payload: Input) -> Answer:
    return ag.llm(
        model=Answer,
        prompt=f"""
        You are Albert Einstein. 
        Answer this Physics question in your personal style:

        Question: {payload.question}
        Data: {payload.data}
        """
    )
```

### 4.2 Adding tooling

You can augment nodes with custom tools. The results of the tools can be injected into the prompts. 
For example, we could define a (contrived) tool that calculates force from mass and acceleration:

```python
@ag.tool
def compute_force(mass: float, acceleration: float) -> Optional[float]:
    try:
        return float(mass) * float(acceleration)
    except Exception:
        return None
```

We can hook this up inside the Newton node, and let the LLM use the tool output to proceed.
(This assumes that the Newton-bot might struggle with the multiplication...)

```python
@ag.step(next="end")
def newton(payload: Input) -> Answer:
    # Here we'll call the compute_force tool that we registered earlier:
    force = None
    if payload.data and "m" in payload.data and "a" in payload.data:
        tr = ag.tool("compute_force", mass=payload.data["m"], acceleration=payload.data["a"])
        force = tr.output if tr.success else None
        ag.emit({
            "kind": "tool",
            "name": "compute_force",
            "input": {"mass": payload.data["m"], "acceleration": payload.data["a"]},
            "output": tr.output,
            "success": tr.success,
        })

    return ag.llm(
        model=Answer,
        prompt=f"""
        You are Isaac Newton. 
        Answer this Physics question in your personal style:

        Question: {payload.question}
        Data: {payload.data}
        Force: {force}
        """
    )

```

### 5. Make an "end" node

```python
@ag.end
def end(payload: Answer) -> Answer:
    # Again, we'll just return the raw payload for now.
    # You could do some final processing here.
    return payload
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
{
  "answer": 15.0,
  "explanation": "According to my laws of motion, F = ma. Given the mass m = 5 kg and acceleration a = 3 m/s^2, we can calculate the net force as F = m * a = 5 * 3 = 15 N."
}
```

## Highlights

- **Declarative Graph Construction**: Define workflow nodes with decorators (`@ag.start`, `@ag.step`, `@ag.route`, `@ag.end`). Each node can prompt an LLM, run custom Python, and more!

- **Automatic Graph Compilation**: The API automatically compiles your code into a `networkx.DiGraph`. This makes it easy to view the graph structure and debug the workflow.

- **Structured Inputs & Outputs**: Robust schema validation using **Pydantic** models.

- **Branching & Routing**: `@ag.route` nodes use LLMs (or custom Python selectors) to choose the next edge. The options are strongly typed.

## Execution Context & Artifacts 

### 1. Execution Context

Each run has a shared context dictionary that persists across nodes. Use if for scratch data, flags, counters, etc., via `ag.vars()`:

Example:

```python
@ag.step(next="read_hint")
def stash_hint(payload: SomeSchema) -> SomeSchema:
    ag.vars()["hint"] = "something"
    return payload

@ag.step(next="end")
def read_hint(payload: SomeSchema) -> SomeSchema:
    hint = ag.vars().get("hint")  
    # do something...
    return payload
```

### 2. Logging with artifacts (via `emit`)

Nodes can produce artifacts: structured events that are appended to the run log. 

```python
ag.emit({"kind":"metrics", "token_usage": 42})
```

### 3. Execution History

When a run finishes, `app.run(...)` returns both the final payload and the execution context (containing history and artifacts).

```python
output, ctx = app.run({"question": "Hi!"})

print(ctx.history)
print(ctx.variables)
```

## Visualization 

It's easy to generate Graphviz visualizations of your agentic workflow.

```python
app.viz(history=ctx.history).save("workflow.png")
```



