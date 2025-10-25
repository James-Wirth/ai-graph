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

### 1. Define your Pydantic schemas

```python
from pydantic import BaseModel

class Question(BaseModel):
    question: str

class SubjectRouter(BaseModel):
    question_type: Literal["physics", "chemistry", ...]

# etc...
```

### 2. Implement a blueprint 

```python
#<my_blueprint_path>/blueprint.py
from aigraph import Blueprint, Message, Context

my_blueprint = Blueprint("my_blueprint")
```

#### e.g. a routing node to classify the subject:
```python
@my_blueprint.node(
    "subject_router",
    emits=["my_blueprint:physics_agent", "my_blueprint:chemistry_agent", ...]
)
def subject_router(msg: Message, ctx: Context) -> List[Message]:
    
    question = Question.model_validate(msg.body)

    # get a structured output from the LLM...
    router_response = ctx.structured(
        model=SubjectRouter,
        prompt=f"Classify this question by subject: {question.question}"
    )
    
    target_node = f"my_blueprint:{router_response.question_type}_agent"
    return [Message(send_to=target_node, body=question)]
```

#### ...and nodes to handle the subject-specific implementation:
```python
@my_blueprint.node("physics_agent")
def physics_agent(msg: Message, ctx: Context) -> List[Message]:

    question = Question.model_validate(msg.body)
    # ... physics-specific logic
    return [...]
```

### 3. Create an app and include the blueprint


```python
from aigraph import App
from <my_blueprint_path>/blueprint import my_blueprint

app = App(name="app")
app.include_blueprint(my_blueprint)

emitted, run_ctx = app.run(
    initial_payload=Question(question="""
        Do the Navier-Stokes equations always have smooth solutions in three dimensions?
    """,
    send_to=["my_blueprint:subject_router"]
)
```
