from aigraph import App, Message, Context

from pydantic import BaseModel


class In(BaseModel):
    text: str


class Out(BaseModel):
    text: str
    touched: bool


def test_pipeline_runs():
    app = App(name="test")

    @app.node("start", emits=["process"])
    def start(msg: Message, ctx: Context):
        payload = In.model_validate(msg.body)
        return Message(send_to="process", body=Out(text=payload.text, touched=False))

    @app.node("process", emits=["finish"])
    def process(msg: Message, ctx: Context):
        payload = Out.model_validate(msg.body)
        return Message(send_to="finish", body=Out(text=payload.text.upper(), touched=True))

    @app.node("finish", emits=["result.v1"])
    def finish(msg: Message, ctx: Context):
        payload = Out.model_validate(msg.body)
        return Message(send_to="result.v1", body=payload)

    emitted, ctx = app.run(initial_payload=In(text="hello"), send_to=["start"])

    result_msg = next((m for m in emitted if m.send_to == "result.v1"), None)
    assert result_msg is not None

    body = result_msg.body
    if hasattr(body, "model_dump"):
        body = body.model_dump()

    assert body["text"] == "HELLO"
    assert body["touched"] is True
    assert len(ctx.history) >= 1
