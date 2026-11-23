"""The chat API end to end in simulated mode: brief, run, approval, answers, safe downloads."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_share_of_voice.api.server import create_app
from ai_share_of_voice.config import load_settings
from ai_share_of_voice.ui.access import AccessGate
from ai_share_of_voice.ui.run_controller import RunController

BRIEF = {
    "brand": "Glovo",
    "competitors": ["Just Eat", "Uber Eats"],
    "sector": "food_delivery",
    "city": "Madrid",
    "country": "España",
    "plan": "quick",
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "simulated")
    monkeypatch.setenv("SEARCH_PROVIDER", "none")
    settings = load_settings()
    controller = RunController(settings, runs_dir=tmp_path, data_dir=Path("data"))
    return TestClient(create_app(settings, controller))


def wait(client, run_id, phases):
    for _ in range(200):
        view = client.get(f"/api/runs/{run_id}").json()
        if view["phase"] not in phases:
            return view
        time.sleep(0.1)
    raise AssertionError(f"run stuck in {view['phase']}")


def test_meta_lists_plans_and_stop_rules(client):
    meta = client.get("/api/meta").json()
    assert meta["mode"] == "simulated"
    assert [p["answers"] for p in meta["plans"].values()] == [10, 30, 100]
    assert meta["stop_rules"]


def test_parse_reads_the_sentence(client):
    parsed = client.post(
        "/api/brief/parse",
        json={"text": "Glovo frente a Just Eat y Uber Eats en Madrid, comida a domicilio"},
    ).json()
    assert parsed["brand"] == "Glovo" and parsed["sector_key"] == "food_delivery"


def test_a_question_naming_a_brand_is_refused(client):
    resp = client.post(
        "/api/runs",
        json={"brief": BRIEF, "questions": ["¿Es Glovo la mejor?", "¿Qué app uso?", "¿Cuál pido?"]},
    )
    assert resp.status_code == 422


def test_full_run(client):
    questions = client.post("/api/brief/questions", json=BRIEF).json()["questions"]
    run_id = client.post("/api/runs", json={"brief": BRIEF, "questions": questions}).json()[
        "run_id"
    ]
    view = wait(client, run_id, {"running"})
    assert view["phase"] == "waiting_approval"
    assert view["funnel"] and view["stop"]["reason_text"]
    assert all("/" not in (e["detail"] or "") for e in view["events"])
    assert client.post(f"/api/runs/{run_id}/decision", json={"approved": True}).status_code == 200
    wait(client, run_id, {"running", "waiting_approval"})
    results = client.get(f"/api/runs/{run_id}/results").json()
    assert results["headline"] and results["brands"]
    reply = client.post(f"/api/runs/{run_id}/ask", json={"question": "¿por qué paraste?"}).json()[
        "reply"
    ]
    assert reply == view["stop"]["reason_text"]
    assert client.get(f"/api/runs/{run_id}/files/client_report.md").status_code == 200
    assert client.get(f"/api/runs/{run_id}/files/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get(f"/api/runs/{run_id}/files/settings.json").status_code == 404


def test_model_calls_need_the_password_when_the_gate_is_on(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "simulated")
    settings = load_settings()
    controller = RunController(settings, runs_dir=tmp_path, data_dir=Path("data"))
    gated = TestClient(create_app(settings, controller, gate=AccessGate("s3cret")))
    assert gated.post("/api/runs", json={"brief": BRIEF}).status_code in (401, 423)
    ok = gated.post("/api/access", headers={"X-Demo-Password": "s3cret"})
    assert ok.status_code == 200
