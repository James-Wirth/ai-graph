<img width="200" alt="image" src="https://github.com/user-attachments/assets/43b3992e-ee0f-4c98-ada9-b2a3d287a969" />

[![status: active development](https://img.shields.io/badge/status-active--development-orange)]()
[![CI](https://github.com/James-Wirth/ai-graph/actions/workflows/ci.yml/badge.svg)](https://github.com/James-Wirth/ai-graph/actions/workflows/ci.yml)


**aiGraph** is a lightweight framework for building graph-based agentic networks.

Nodes are decorated python functions (with a Flask-like syntax) which consume/emit Pydantic schemas, making it easy to build complex workflows (e.g. branching, loops, fan-ins/outs). 

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

## Example workflow

### 1. Define a blueprint with nodes

```python
#<some_path>/demo.py
from pydantic import BaseModel

from aigraph import Blueprint, Message, Context

my_blueprint = Blueprint("my_blueprint")

@my_blueprint.node("alice", emits=["my_blueprint:bob", ...])
def alice(msg: Message, ctx: Context) -> List[Message]:

    class AliceIn(BaseModel): question: str
    _in = AliceIn.model_validate(msg.body)

    class AliceOut(BaseModel): answer: str
    _out_ = ctx.structured(model=AliceOut, prompt=f"Answer briefly: {_in.question}")
    
    return [
      Message(send_to="my_blueprint:bob", body=_out),
      ...
    ]

@my_blueprint.node("bob", emits=...)
def bob(msg: Message, ctx: Context) -> List[Message]:
    ...
```

### 2. Create an app and include the blueprint


```python
from aigraph import App
from <some_path>.demo import my_blueprint

app = App(name="app")
app.include_blueprint(my_blueprint)

initial_payload = AliceIn(question="What's the meaning of life?")
emitted, ctx_run = app.run(initial_payload, send_to=["my_blueprint:alice"])
print("Final emitted types:", [m.type for m in emitted])
```
