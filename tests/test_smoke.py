from pydantic import BaseModel

import aigraph as ag


class In(BaseModel):
    text: str


class Out(BaseModel):
    text: str
    touched: bool


@ag.start(next="process")
def start(payload: In) -> Out:
    return Out(text=payload.text, touched=False)


@ag.step(next="finish")
def process(payload: Out) -> Out:
    return Out(text=payload.text.upper(), touched=True)


@ag.end
def finish(payload: Out) -> Out:
    return payload


def test_pipeline_runs():
    app = ag.App()
    result, ctx = app.run({"text": "hello"})
    assert result["text"] == "HELLO"
    assert result["touched"] is True
    assert len(ctx.history) >= 1
