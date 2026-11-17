"""The control app (mlops/web): the login, the pages it serves, the Editor's routes, the pipeline graph, the
monitoring and what it refuses (paths out of their folder, figures that are not PNG or PDF)."""

import base64

import pytest
from conftest import make_page, make_zip
from fastapi.testclient import TestClient


def auth(user="ana", password="pw"):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


@pytest.fixture
def client(env, monkeypatch):
    monkeypatch.setenv("MLOPS_ADMINS", "ana")
    from mlops.web.api import create_app
    with TestClient(create_app({"ana": "pw", "bo": "pw2"}, sample=False)) as c:
        c.headers.update(auth())
        yield c


def ingest(env, pages=6):
    from mlops import db, ingest as ingesting
    source = env / "export"
    for n in range(pages):
        make_page(source, f"03010108_1_1_I_20{10 + n % 3}_{n + 1}-2-1", (n * 20, 90, 120))
    with db.connection() as conn:
        batch = db.one(conn, "INSERT INTO batches (status) VALUES ('uploaded') RETURNING id")
        ingesting.ingest(conn, batch["id"], make_zip(source, env / "export.zip"), log=lambda *_: None)
        return [r["id"] for r in db.all_rows(conn, "SELECT id FROM pages ORDER BY id")]


def test_a_login_is_asked_everywhere_but_the_health_check(client):
    anonymous = TestClient(client.app)
    assert anonymous.get("/healthz").json() == {"status": "ok"}
    for url in ("/", "/api/meta", "/api/pipeline", "/metrics", "/static/app.js"):
        answer = anonymous.get(url)
        assert answer.status_code == 401 and "Basic" in answer.headers["www-authenticate"], url
    assert anonymous.get("/api/meta", headers=auth("ana", "wrong")).status_code == 401
    assert anonymous.get("/api/meta", headers=auth("bo", "pw2")).json()["admin"] is False
    page = client.get("/")
    assert page.status_code == 200 and 'x-data="mlops()"' in page.text
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    assert client.get("/api/meta").headers["cache-control"] == "no-store"
    with pytest.raises(SystemExit):
        from mlops.web.api import create_app
        create_app({})


def test_the_pages_of_the_app_answer(client, env):
    ingest(env)
    for url in ("/api/meta", "/api/data", "/api/data/vocabulary", "/api/data/writers", "/api/data/split",
                "/api/data/normalization", "/api/data/augmentation", "/api/data/missing", "/api/data/preview",
                "/api/training", "/api/evaluation", "/api/experiments", "/api/jobs", "/api/monitor?minutes=5",
                "/api/pages?split=train", "/api/pages?split=augmented", "/static/index.html", "/static/editor.js"):
        assert client.get(url).status_code == 200, url
    data = client.get("/api/data").json()
    assert data["total"]["pages"] == 6 and len(data["digest"]) > 6
    assert client.get("/api/jobs?status=nonsense").status_code == 400
    assert client.get("/api/monitor?minutes=500").status_code == 400
    assert client.get("/api/pages?split=elsewhere").status_code == 400


def test_the_pipeline_graph_and_the_metrics(client, env):
    ingest(env)
    graph = client.get("/api/pipeline").json()
    ids = {n["id"] for n in graph["nodes"]}
    assert {"export", "ingest", "edit", "version", "train_htr", "evaluate_blla", "release", "madoc"} <= ids
    assert all(e["from"] in ids and e["to"] in ids for e in graph["edges"])
    client.post("/api/training/htr", json={})
    graph = client.get("/api/pipeline").json()
    assert next(n for n in graph["nodes"] if n["id"] == "train_htr")["state"] == "queued"
    assert [j["kind"] for j in graph["active"]] == ["train"]
    metrics = client.get("/metrics").text
    assert "mlops_uptime_seconds" in metrics and 'mlops_jobs_total{kind="train",task="htr",status="all"} 1' in metrics


def test_the_editor_saves_rotates_and_restores_a_page(client, env):
    page_id = ingest(env)[0]
    listing = client.get("/api/pages?split=all&search=" + page_id[:8]).json()
    assert page_id in [p["id"] for p in listing["pages"]]
    page = client.get(f"/api/pages/{page_id}").json()
    assert page["revision"] == 0 and page["history"] == [] and len(page["lines"]) == 6
    image = client.get(f"/api/pages/{page_id}/image?r=0")
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/pages/{page_id}/xml").content.startswith(b"<?xml")

    lines = page["lines"]
    lines[0]["text"] = "سطر  مصحح"
    answer = client.put(f"/api/pages/{page_id}/lines", json={"lines": lines[:5], "revision": 0}).json()
    assert answer["result"]["revision"] == 1 and len(answer["page"]["lines"]) == 5
    assert answer["page"]["lines"][0]["text"] == "سطر مصحح"
    assert [h["username"] for h in answer["page"]["history"]] == ["ana", "ana"]
    # a stale revision is a conflict; a bad body a refusal
    assert client.put(f"/api/pages/{page_id}/lines", json={"lines": lines, "revision": 0}).status_code == 409
    assert client.put(f"/api/pages/{page_id}/lines", json={"lines": "x", "revision": 1}).status_code == 400
    rotated = client.post(f"/api/pages/{page_id}/transform", json={"op": {"op": "rotate", "degrees": 90},
                                                                   "revision": 1}).json()["page"]
    assert (rotated["width"], rotated["height"], rotated["revision"]) == (300, 200, 2)
    assert client.post(f"/api/pages/{page_id}/transform",
                       json={"op": {"op": "rotate", "degrees": 33}, "revision": 2}).status_code == 400
    restored = client.post(f"/api/pages/{page_id}/restore", json={"restore": 0, "revision": 2}).json()["page"]
    assert (restored["width"], restored["revision"], len(restored["lines"])) == (200, 3, 6)
    assert client.post(f"/api/pages/{page_id}/restore", json={"restore": "x", "revision": 3}).status_code == 400
    assert client.get("/api/pages?split=all&edited=true").json()["pages"][0]["revision"] == 3
    assert client.get("/api/data").json()["total"]["edited"] == 1


def test_a_page_or_a_file_out_of_its_folder_is_not_found(client, env):
    ingest(env)
    for url in ("/api/pages/nope", "/api/pages/nope/image", "/api/pages/..%2F..%2Fetc%2Fpasswd/xml",
                "/api/pages/aug:..:passwd", "/api/pages/aug:x:..%2F..%2Fsecret/image",
                "/api/figures/model/1/x.png", "/api/figures/model/1/..%2Fsecret.png", "/api/figures/table/1/x.png",
                "/api/figures/model/1/notes.txt", "/static/../api.py"):
        assert client.get(url).status_code == 404, url
    assert client.put("/api/pages/nope/lines", json={"lines": [], "revision": 0}).status_code == 404


def test_augmented_pages_are_listed_read_only(client, env):
    from mlops.config import path
    page_id = ingest(env)[0]
    folder = path("data") / "augmented" / "abcdef-c1-s42"
    make_page(folder, f"{page_id}_aug1", (10, 10, 10))
    (folder / ".done").write_text("")
    listing = client.get("/api/pages?split=augmented").json()["pages"]
    assert [p["key"] for p in listing] == [f"aug:abcdef-c1-s42:{page_id}_aug1"]
    page = client.get(f'/api/pages/{listing[0]["key"]}').json()
    assert page["readonly"] and len(page["lines"]) == 6
    assert client.get(f'/api/pages/{listing[0]["key"]}/image').status_code == 200


def test_the_preview_download_is_for_admins_only(client, env):
    assert client.get("/api/data/preview/download", headers=auth("bo", "pw2")).status_code == 403
    assert client.get("/api/data/preview/download").status_code == 400      # an admin, but no preview yet


def test_jobs_are_queued_listed_and_cancelled(client, env):
    ingest(env)
    queued = client.post("/api/data/augmentation", json={"copies": 2, "seed": 1}).json()
    assert client.post("/api/data/augmentation", json={"copies": 50}).status_code == 400
    job = client.get(f'/api/jobs/{queued["job"]}').json()
    assert job["kind"] == "augment" and job["status"] == "queued" and job["args"]["copies"] == 2
    assert client.post(f'/api/jobs/{queued["job"]}/cancel').status_code == 200
    assert client.get("/api/jobs?status=cancelled").json()["jobs"][0]["id"] == queued["job"]
    assert client.post("/api/jobs/99999/cancel").status_code == 404
    assert client.post("/api/experiments", json={"recognizer": "other"}).status_code == 400
    assert client.post("/api/experiments", json={"recognizer": "crnn", "options": "{bad"}).status_code == 400
