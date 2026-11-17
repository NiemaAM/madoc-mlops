"""The HTTP routes of the control app (FastAPI). Every change goes through the modules of the platform; a long
task is always a job of the queue (the worker runs it), so the app can restart at any time.

    GET  /                the interface (static/index.html: one page, no build step)
    GET  /healthz         the app and its database answer (no login: Docker's healthcheck)
    GET  /metrics         the resources and the jobs for Prometheus
    /api/...              the data of the pages, and the actions

A login is always asked (HTTP Basic, MLOPS_AUTH="name:password,..."); MLOPS_ADMINS names who may download the
preview of the dataset (personal data; every download is recorded). The answers of /api are never cached.
"""

import base64
import json
import os
import secrets
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from mlops import db, editing, jobs, settings as app_settings
from mlops.config import SPLITS, TASKS, params, path
from mlops.web import monitor, views

STATIC = Path(__file__).resolve().parent / "static"
PUBLIC = ("/healthz",)
RECOGNIZERS = ("kraken", "crnn", "trocr", "qwen")


class Refused(Exception):
    """An action that cannot be done now, with a message for the person (HTTP 400, or 409 for a conflict)."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def logins(text=None):
    pairs = {}
    for item in (os.environ.get("MLOPS_AUTH", "") if text is None else text).split(","):
        user, _, password = item.strip().partition(":")
        if user and password:
            pairs[user] = password
    return pairs


def admins(users):
    names = [n.strip() for n in os.environ.get("MLOPS_ADMINS", "").split(",") if n.strip()]
    return names or list(users)[:1]


def user_of(header, users):
    """The name of the person of this Authorization header, or None (every login compared in constant time)."""
    if not header.lower().startswith("basic "):
        return None
    try:
        user, _, password = base64.b64decode(header[6:]).decode("utf-8").partition(":")
    except (ValueError, UnicodeDecodeError):
        return None
    found = None
    for known, secret in users.items():
        if secrets.compare_digest(user.encode(), known.encode()) & secrets.compare_digest(password.encode(),
                                                                                        secret.encode()):
            found = known
    return found


def connection():
    return db.connection()


def queue(kind, task=None, **args):
    with connection() as conn:
        job = jobs.enqueue(conn, kind, task, **args)
    return {"job": job["id"], "message": f'Job {job["id"]} queued: {kind}{" " + task if task else ""}.'}


def busy(conn, kinds):
    return db.one(conn, "SELECT id, kind FROM jobs WHERE status IN ('queued', 'running') AND kind = ANY(%s) "
                        "ORDER BY id LIMIT 1", list(kinds))


def no_store(data):
    return JSONResponse(data, headers={"Cache-Control": "no-store"})


def create_app(users=None, sample=True):
    users = logins() if users is None else dict(users)
    if not users:
        raise SystemExit("MLOPS_AUTH is not set (.env): the app always asks for a login")
    admin_names = admins(users)

    @asynccontextmanager
    async def lifespan(app):
        db.init_schema()
        remove_old_uploads()
        if sample:
            monitor.start()
        yield

    app = FastAPI(title="MaDoc MLOps", docs_url=None, redoc_url=None, openapi_url="/api/openapi.json",
                  lifespan=lifespan)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.url.path not in PUBLIC:
            user = user_of(request.headers.get("authorization", ""), users)
            if not user:
                return Response("Login required.", status_code=401,
                                headers={"WWW-Authenticate": 'Basic realm="MaDoc MLOps", charset="UTF-8"'})
            request.state.user = user
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' blob: data:; script-src 'self' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(Refused)
    async def refused(request, error):
        return JSONResponse({"detail": str(error)}, status_code=error.status)

    @app.exception_handler(editing.Conflict)
    async def conflict(request, error):
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.exception_handler(editing.EditError)
    async def edit_error(request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.exception_handler(editing.PageNotFound)
    @app.exception_handler(FileNotFoundError)
    async def not_found(request, error):
        return JSONResponse({"detail": "not found"}, status_code=404)

    def guarded(function, *args, **kwargs):
        """A module's ValueError is a message for the person."""
        try:
            return function(*args, **kwargs)
        except (editing.EditError, editing.PageNotFound, FileNotFoundError):
            raise
        except ValueError as e:
            raise Refused(str(e)) from None

    # -- the interface ---------------------------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/healthz", include_in_schema=False)
    def health():
        with connection() as conn:
            db.one(conn, "SELECT 1 AS ok")
        return {"status": "ok"}

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        with connection() as conn:
            stats = views.job_stats(conn, days=30)
        return PlainTextResponse(monitor.prometheus(stats))

    @app.get("/api/meta")
    def meta(request: Request):
        from mlops.browse import type_choices
        return {"user": request.state.user, "admin": request.state.user in admin_names, "splits": list(SPLITS),
                "tasks": [{"id": t, "name": views.TASK_NAMES[t]} for t in TASKS],
                "page_types": [{"value": v, "label": label} for label, v in type_choices()],
                "max_upload_mb": params()["ingest"]["max_upload_mb"], "reading_jobs": list(editing.READING_JOBS)}

    @app.get("/api/pipeline")
    def pipeline():
        with connection() as conn:
            return views.pipeline(conn)

    # -- data ------------------------------------------------------------------------------------------------
    @app.get("/api/data")
    def data():
        with connection() as conn:
            return {**views.dataset(conn), "versions": views.versions(conn), "batches": views.batches(conn)}

    @app.post("/api/data/upload")
    async def upload(request: Request, file: UploadFile = File(...), held_out: bool = Form(False)):
        limit = params()["ingest"]["max_upload_mb"] * 2 ** 20
        if not (file.filename or "").lower().endswith(".zip"):
            raise Refused("A .zip file is expected (the eScriptorium export: PageXML with images).")
        with connection() as conn:
            batch = db.one(conn, "INSERT INTO batches (status, held_out) VALUES ('uploaded', %s) RETURNING id",
                           bool(held_out))
        target = path("data") / "uploads" / f'batch-{batch["id"]}.zip'
        target.parent.mkdir(parents=True, exist_ok=True)
        partial, size = target.with_suffix(".part"), 0
        try:
            with open(partial, "wb") as out:
                while chunk := await file.read(2 ** 20):
                    size += len(chunk)
                    if size > limit:
                        raise Refused(f'The zip is larger than {params()["ingest"]["max_upload_mb"]} MB.')
                    out.write(chunk)
            partial.replace(target)
        except BaseException:
            partial.unlink(missing_ok=True)
            with connection() as conn:
                conn.execute("UPDATE batches SET status = 'failed', report = %s WHERE id = %s",
                             (db.json({"reasons": {"upload": 1}}), batch["id"]))
            raise
        with connection() as conn:
            job = jobs.enqueue(conn, "ingest", batch_id=batch["id"], upload=str(target))
        return {"batch": batch["id"], "job": job["id"],
                "message": f'Batch {batch["id"]} received{" (held-out set: always in test)" if held_out else ""}: '
                           f'ingestion queued (job {job["id"]}).'}

    @app.post("/api/data/batches/{batch_id}/{action}")
    def batch_action(batch_id: int, action: str, body: dict = Body(default={})):
        from mlops import batches
        with connection() as conn:
            if action == "delete":
                n = guarded(batches.delete, conn, batch_id)
                return {"message": f"Batch {batch_id}: {n} pages taken out of the dataset (restore brings them back)."}
            if action == "restore":
                n = guarded(batches.restore, conn, batch_id)
                return {"message": f"Batch {batch_id}: {n} pages back in the dataset."}
            if action == "purge":
                n = guarded(batches.purge, conn, batch_id, str(body.get("confirmation") or ""))
                return {"message": f"Batch {batch_id}: {n} pages removed for good."}
        raise HTTPException(404)

    @app.get("/api/data/vocabulary")
    def vocabulary():
        with connection() as conn:
            return views.vocabulary(conn)

    @app.get("/api/data/writers")
    def writers():
        with connection() as conn:
            return views.writers(conn)

    @app.post("/api/data/writers/pin")
    def pin(body: dict = Body(...)):
        from mlops import writers as w
        with connection() as conn:
            guarded(w.pin, conn, str(body.get("writer") or ""), str(body.get("split") or "auto"))
            return views.writers(conn)

    @app.post("/api/data/writers/alias")
    def alias(body: dict = Body(...)):
        from mlops import writers as w
        with connection() as conn:
            guarded(w.add_alias, conn, str(body.get("variant") or ""), str(body.get("canonical") or ""))
            return views.writers(conn)

    @app.delete("/api/data/writers/alias")
    def unalias(body: dict = Body(...)):
        from mlops import writers as w
        with connection() as conn:
            w.remove_alias(conn, str(body.get("variant") or ""))
            return views.writers(conn)

    @app.get("/api/data/split")
    def split_values():
        with connection() as conn:
            return views.split_settings(conn)

    @app.post("/api/data/split/{action}")
    def split(action: str, body: dict = Body(default={})):
        from mlops.ingest import resplit
        with connection() as conn:
            if action == "reset":
                app_settings.reset(conn, "split")
                return {"settings": views.split_settings(conn), "message": "Back to the settings of params.yaml."}
            value = guarded(views.split_from, body)
            if action == "preview":
                return {"result": views.split_result(resplit(conn, log=lambda *_: None, split=value, apply=False))}
            if action == "apply":
                if busy(conn, ("ingest", "normalize")):
                    raise Refused("An ingest or a normalisation is queued or running: split again once it is done.", 409)
                app_settings.save(conn, "split", value)
                result = resplit(conn, log=lambda *_: None, split=value, apply=True)
                return {"result": views.split_result(result), "settings": views.split_settings(conn),
                        "message": "Split applied: " + ", ".join(f"{s} {n} pages" for s, n in result["pages"].items())}
        raise HTTPException(404)

    @app.get("/api/data/normalization")
    def normalization():
        with connection() as conn:
            return {"rules": views.normalization(conn)}

    @app.post("/api/data/normalization/{action}")
    def normalization_action(action: str, body: dict = Body(default={})):
        from mlops import normalize
        from mlops.ingest import normalize_dataset
        with connection() as conn:
            if action in ("add", "remove"):
                source = body.get("source")
                if not isinstance(source, str) or (not source.strip() and source != " "):
                    raise Refused("Type the character or the text to replace (or its code, e.g. U+0649).")
                if action == "add":
                    guarded(normalize.add_rule, conn, source, str(body.get("replacement") or ""))
                else:
                    guarded(normalize.remove_rule, conn, source)
                return {"rules": views.normalization(conn)}
            if action == "preview":
                result = normalize_dataset(conn, log=lambda *_: None, apply=False)
                return {"pages": result["pages"], "replaced": result["replaced"],
                        "text": f"{result['pages']} pages would change: {normalize.describe(result['replaced'])}"}
            if action == "run":
                if busy(conn, ("ingest", "normalize")):
                    raise Refused("An ingestion or a normalisation is already queued or running.", 409)
        if action == "run":
            return queue("normalize")
        raise HTTPException(404)

    @app.get("/api/data/augmentation")
    def augmentation():
        with connection() as conn:
            return views.augmentations(conn)

    @app.post("/api/data/augmentation")
    def augment(body: dict = Body(...)):
        try:
            copies, seed = int(body.get("copies") or 1), int(body.get("seed") or 0)
        except (TypeError, ValueError):
            raise Refused("Copies and seed: whole numbers.") from None
        if not 1 <= copies <= 10 or seed < 0:
            raise Refused("Copies: 1 to 10; seed: 0 or more.")
        return queue("augment", copies=copies, seed=seed)

    @app.get("/api/data/missing")
    def missing():
        with connection() as conn:
            return {"rows": views.missing(conn)}

    @app.get("/api/data/missing/download")
    def missing_download():
        from mlops.missing import listing
        with connection() as conn:
            text = listing(conn)
        if not text:
            raise Refused("No act is missing a page.")
        return Response(text, media_type="text/plain; charset=utf-8",
                        headers={"Content-Disposition": "attachment; filename=pages_to_annotate.txt"})

    @app.get("/api/data/preview")
    def preview(request: Request):
        return {"latest": views.preview(), "admin": request.state.user in admin_names}

    @app.post("/api/data/preview")
    def preview_build():
        return queue("preview")

    @app.get("/api/data/preview/download")
    def preview_download(request: Request):
        from mlops.preview import latest, record_download
        if request.state.user not in admin_names:
            raise Refused("Only an admin of the app may download the preview (MLOPS_ADMINS).", 403)
        file = latest()
        if not file:
            raise Refused("Make the preview first.")
        with connection() as conn:
            record_download(conn, request.state.user, file.name)
        return FileResponse(file, filename=file.name, media_type="application/zip")

    # -- training --------------------------------------------------------------------------------------------
    @app.get("/api/training")
    def training():
        with connection() as conn:
            return {"form": views.training_form(conn), "models": views.models(conn)}

    @app.post("/api/training/{task}")
    def train(task: str, body: dict = Body(...)):
        if task not in TASKS:
            raise HTTPException(404)
        options = guarded(app_settings.clean, task, body)
        with connection() as conn:
            app_settings.save(conn, f"train.{task}", options)
        return queue("train", task, settings=options, use_augmented=bool(options.get("use_augmented")))

    @app.delete("/api/training/{task}")
    def reset_training(task: str):
        if task not in TASKS:
            raise HTTPException(404)
        with connection() as conn:
            app_settings.reset(conn, f"train.{task}")
            return {"form": views.training_form(conn)[task], "message": f"{views.TASK_NAMES[task]}: back to params.yaml."}

    @app.get("/api/models/{model_id}")
    def model(model_id: int):
        with connection() as conn:
            row = db.one(conn, "SELECT figures FROM models WHERE id = %s", model_id)
        if not row:
            raise HTTPException(404)
        return {"figures": views.figures(row["figures"])}

    # -- evaluation, release ---------------------------------------------------------------------------------
    @app.get("/api/evaluation")
    def evaluation_page():
        with connection() as conn:
            return {"evaluations": views.evaluations(conn), "models": views.models(conn),
                    "options": {t: views.model_options(conn, t) for t in TASKS}}

    @app.get("/api/evaluations/{evaluation_id}")
    def evaluation(evaluation_id: int):
        with connection() as conn:
            found = views.evaluation(conn, evaluation_id)
        if not found:
            raise HTTPException(404)
        return found

    @app.post("/api/evaluate")
    def evaluate(body: dict = Body(...)):
        with connection() as conn:
            model = db.one(conn, "SELECT id, task FROM models WHERE id = %s", int(body.get("model_id") or 0))
        if not model:
            raise Refused("Choose a candidate model.")
        return queue("evaluate", model["task"], model_id=model["id"])

    @app.post("/api/evaluate/pipeline")
    def evaluate_pipeline(body: dict = Body(...)):
        chosen = {}
        with connection() as conn:
            for key, task in (("segmentation", "blla"), ("recognition", "htr")):
                value = str(body.get(key) or "production")
                if value != "production" and not db.one(conn, "SELECT 1 AS x FROM models WHERE id = %s AND task = %s",
                                                        int(value) if value.isdigit() else -1, task):
                    raise Refused(f"{key}: production or a {task} model of the table.")
                chosen[key] = value
        return queue("pipeline", **chosen)

    @app.post("/api/evaluations/{evaluation_id}/release")
    def release(evaluation_id: int, body: dict = Body(default={})):
        with connection() as conn:
            e = db.one(conn, "SELECT task, passed, pr_url FROM evaluations WHERE id = %s", evaluation_id)
            if not e:
                raise HTTPException(404)
            if e["pr_url"]:
                raise Refused(f'A pull request is already open: {e["pr_url"]}', 409)
            if busy(conn, ("release",)):
                raise Refused("A release is already queued or running.", 409)
        override = bool(body.get("override"))
        if not e["passed"] and not override:
            raise Refused("The candidate is not better: confirm \"Open anyway\" to open the pull request regardless.")
        return queue("release", e["task"], evaluation_id=evaluation_id, override=override)

    # -- experiments -----------------------------------------------------------------------------------------
    @app.get("/api/experiments")
    def experiments():
        with connection() as conn:
            return {"experiments": views.experiments(conn), "seed": params()["seed"]}

    @app.get("/api/experiments/{experiment_id}")
    def experiment(experiment_id: int):
        with connection() as conn:
            found = views.experiment(conn, experiment_id)
        if not found:
            raise HTTPException(404)
        return found

    @app.post("/api/experiments")
    def run_experiment(body: dict = Body(...)):
        recognizer = body.get("recognizer", "kraken")
        scheme = body.get("scheme", "writers")
        if recognizer not in RECOGNIZERS or scheme not in ("writers", "split"):
            raise Refused("Recognizer: kraken, crnn, trocr or qwen; pages: writers or split.")
        try:
            seeds = [int(x) for x in str(body.get("seeds") or params()["seed"]).replace(" ", "").split(",") if x]
        except ValueError:
            raise Refused("Seeds: whole numbers separated by commas (e.g. 1,2,3).") from None
        if not seeds or len(seeds) > 10:
            raise Refused("Seeds: one to ten.")
        options = body.get("options") or {}
        if isinstance(options, str):
            try:
                options = json.loads(options) if options.strip() else {}
            except ValueError as e:
                raise Refused(f"Options: not JSON ({e}).") from None
        if not isinstance(options, dict):
            raise Refused("Options: a JSON object, e.g. {\"epochs\": 100}.")
        args = {"recognizer": recognizer, "scheme": scheme, "seeds": seeds, "segmentation": bool(body.get("segmentation"))}
        if recognizer == "kraken":
            with connection() as conn:
                args["settings"] = {**app_settings.training(conn, "htr"), **options}
                args["blla_settings"] = app_settings.training(conn, "blla")
        else:
            args["settings"] = options
        return queue("crossval", None, **args)

    # -- jobs ------------------------------------------------------------------------------------------------
    @app.get("/api/jobs")
    def job_list(status: str = "", kind: str = "", limit: int = 100):
        if status and status not in ("queued", "running", "done", "failed", "cancelled"):
            raise Refused("unknown status")
        if kind and kind not in jobs.KINDS:
            raise Refused("unknown kind")
        with connection() as conn:
            return {"jobs": views.jobs(conn, min(max(limit, 1), 500), status or None, kind or None),
                    "kinds": list(jobs.KINDS), "workers": views.workers(conn),
                    "paused": app_settings.paused(conn)}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: int):
        with connection() as conn:
            found = views.job(conn, job_id)
        if not found:
            raise HTTPException(404)
        return found

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: int):
        with connection() as conn:
            if not db.one(conn, "SELECT 1 AS x FROM jobs WHERE id = %s", job_id):
                raise HTTPException(404)
            jobs.cancel(conn, job_id)
        return {"message": f"Job {job_id}: cancel requested."}

    # -- figures ---------------------------------------------------------------------------------------------
    @app.get("/api/figures/{kind}/{owner}/{name}")
    def figure(kind: str, owner: int, name: str):
        """One PNG or PDF of the figures folder of a model, an evaluation or an experiment, by its exact name."""
        table = {"model": "models", "evaluation": "evaluations", "experiment": "experiments"}.get(kind)
        if not table or Path(name).name != name or Path(name).suffix not in (".png", ".pdf"):
            raise HTTPException(404)
        with connection() as conn:
            row = db.one(conn, f"SELECT figures FROM {table} WHERE id = %s", owner)
        folder = Path(row["figures"]) if row and row["figures"] else None
        if not folder or not (folder / name).is_file():
            raise HTTPException(404)
        return FileResponse(folder / name, headers={"Cache-Control": "private, max-age=3600"})

    # -- the pages (Editor) ----------------------------------------------------------------------------------
    @app.get("/api/pages")
    def pages(split: str = "train", type: str = "all", names: bool = False, search: str = "", edited: bool = False):
        if split not in (*SPLITS, views.AUGMENTED, "all"):
            raise Refused("unknown split")
        with connection() as conn:
            return views.page_list(conn, split, type or "all", names, search, edited)

    @app.get("/api/pages/{page_id}")
    def page(page_id: str, names: bool = False):
        if page_id.startswith("aug:"):
            return views.augmented_detail(page_id)
        with connection() as conn:
            return views.page_detail(conn, page_id, names)

    @app.get("/api/pages/{page_id}/image")
    def page_image(page_id: str):
        if page_id.startswith("aug:"):
            _, image = views.augmented_files(page_id)
        else:
            with connection() as conn:
                editing.page_row(conn, page_id)
            _, image = editing.files_of(page_id)
        if not image.is_file():
            raise HTTPException(404)
        # the address carries the revision (?r=): a new revision is never shown from the browser's cache
        return FileResponse(image, headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/pages/{page_id}/xml")
    def page_xml(page_id: str):
        xml = views.augmented_files(page_id)[0] if page_id.startswith("aug:") else editing.files_of(page_id)[0]
        return Response(xml.read_bytes(), media_type="application/xml",
                        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(xml.name)}"})

    @app.put("/api/pages/{page_id}/lines")
    def save_lines(request: Request, page_id: str, body: dict = Body(...)):
        with connection() as conn:
            result = editing.save_lines(conn, page_id, body.get("lines"), body.get("revision"), request.state.user)
            return {"result": result, "page": views.page_detail(conn, page_id, bool(body.get("names")))}

    @app.post("/api/pages/{page_id}/transform")
    def transform(request: Request, page_id: str, body: dict = Body(...)):
        with connection() as conn:
            result = editing.transform(conn, page_id, body.get("op"), body.get("revision"), request.state.user)
            return {"result": result, "page": views.page_detail(conn, page_id, bool(body.get("names")))}

    @app.post("/api/pages/{page_id}/restore")
    def restore(request: Request, page_id: str, body: dict = Body(...)):
        try:
            revision = int(body.get("restore"))
        except (TypeError, ValueError):
            raise Refused("restore: the number of a revision") from None
        with connection() as conn:
            result = editing.restore(conn, page_id, revision, body.get("revision"), request.state.user)
            return {"result": result, "page": views.page_detail(conn, page_id, bool(body.get("names")))}

    # -- monitoring ------------------------------------------------------------------------------------------
    @app.get("/api/monitor")
    def monitoring(minutes: int = 15):
        if not 1 <= minutes <= 60:
            raise Refused("minutes: 1 to 60")
        with connection() as conn:
            return {**monitor.snapshot(minutes), "workers": views.workers(conn), "stats": views.job_stats(conn),
                    "paused": app_settings.paused(conn),
                    "running": views.jobs(conn, 10, "running", None) + views.jobs(conn, 20, "queued", None)}

    return app


def remove_old_uploads():
    """Uploads left half written by a stopped app (".part")."""
    folder = path("data") / "uploads"
    for file in folder.glob("*.part") if folder.is_dir() else []:
        file.unlink(missing_ok=True)
    shutil.rmtree(folder / "tmp", ignore_errors=True)
