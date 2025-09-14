import aigraph as ag

from pydantic import BaseModel


class In(BaseModel):
    text: str


class Out(BaseModel):
    text: str
    touched: bool


def test_pipeline_runs():
    app = ag.App(name="test")

    @app.start(next="process")
    def start(payload: In) -> Out:
        return Out(text=payload.text, touched=False)

    @app.step(next="finish")
    def process(payload: Out) -> Out:
        return Out(text=payload.text.upper(), touched=True)

    @app.end()
    def finish(payload: Out) -> Out:
        return payload

    result, ctx = app.run({"text": "hello"})
    assert result["text"] == "HELLO"
    assert result["touched"] is True
    assert len(ctx.history) >= 1
