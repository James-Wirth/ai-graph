# aiGraph

[![status: active development](https://img.shields.io/badge/status-active--development-orange)]()

**aiGraph** is a lightweight framework for building graph-based agentic networks.

Nodes are decorated python functions that consume/emit typed messages (via Pydantic models), making it easy to build complex workflows (e.g. branching, loops, fan-ins/outs). The LLM interfaces, inventory, tools, artifacts (etc.) are carried around by an explicit context.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8dae2ebd-8742-44ec-858a-553067c6a6b8" width="500" />
</p>

## Usage

This library is under active development. 
Expect frequent changes and breaking updates!

```
git clone https://github.com/James-Wirth/ai-graph.git
cd aigraph
pip install -e ".[dev]"  
```

## Example node

```python
import aigraph.api as ag

from aigraph.core.messages import Message
from aigraph.core.context import Context

from pydantic import BaseModel

app = ag.App(name="example")

class AliceIn(BaseModel): question: str
class AliceOut(BaseModel): answer: str

@app.node("alice", emits=["bob"])
def alice(msg: Message, ctx: Context) -> Message:
    _in = AliceIn.model_validate(msg.body)
    _out_ = ctx.structured(model=AliceOut, prompt=f"Answer briefly: {_in.question}")
    return Message(type="bob", body=_out)

@app.node("bob", emits=...)
def bob(msg: Message, ctx: Context) -> Message:
    ...
```

Run the graph by specifying the initial payload and the seed node:

```python
initial_payload = AliceIn(question="What's the meaning of life?")
emitted, ctx_run = app.run(initial_payload, seed_type=["alice"])
```
