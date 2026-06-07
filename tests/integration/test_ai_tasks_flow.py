import pytest

pytestmark = pytest.mark.integration


class TestFreeFormEndpoints:
    def test_ask_echoes_through_real_service(self, ai_tasks_harness):
        resp = ai_tasks_harness.client.post("/ask", json={"question": "hello there"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "echo: hello there"
        assert body["model"] == "fake-llm-model"
        assert body["tokens_used"] > 0

    def test_summarize(self, ai_tasks_harness):
        resp = ai_tasks_harness.client.post(
            "/summarize", json={"text": "some long text to summarize"}
        )
        assert resp.status_code == 200
        assert resp.json()["summary"].startswith("echo:")

    def test_translate_includes_target_language_prompt(self, ai_tasks_harness):
        resp = ai_tasks_harness.client.post(
            "/translate", json={"text": "hello", "target_language": "Spanish"}
        )
        assert resp.status_code == 200
        assert resp.json()["translation"] == "echo: hello"
        assert ai_tasks_harness.llm.calls[-1]["endpoint"] == "translate"

    def test_ask_stream_aggregates_tokens(self, ai_tasks_harness):
        resp = ai_tasks_harness.client.post("/ask-stream", json={"question": "hi"})
        assert resp.status_code == 200
        assert resp.text == "Hello world"


class TestStructuredEndpoints:
    def test_classify_parses_and_validates_json(self, ai_tasks_harness):
        resp = ai_tasks_harness.client.post("/classify", json={"text": "great product"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["sentiment"] == "positive"
        assert body["keywords"] == ["great", "product"]

    def test_extract_keywords(self, ai_tasks_harness):
        resp = ai_tasks_harness.client.post(
            "/extract-keywords", json={"text": "alpha beta gamma delta"}
        )
        assert resp.status_code == 200
        assert resp.json()["keywords"] == ["alpha", "beta", "gamma"]

    def test_analyze_text_combines_fields(self, ai_tasks_harness):
        resp = ai_tasks_harness.client.post(
            "/analyze-text", json={"text": "neutral observation about weather"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["language"] == "en"
        assert body["sentiment"] == "neutral"
        assert body["prompt_version"]


class TestRequestPlumbing:
    def test_request_id_header_round_trips(self, ai_tasks_harness):
        resp = ai_tasks_harness.client.post(
            "/ask", json={"question": "hi"}, headers={"x-request-id": "trace-xyz"}
        )
        assert resp.headers["x-request-id"] == "trace-xyz"

    def test_validation_rejects_empty_input(self, ai_tasks_harness):
        resp = ai_tasks_harness.client.post("/ask", json={"question": ""})
        assert resp.status_code == 422
