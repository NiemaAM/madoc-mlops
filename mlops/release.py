"""The models in production and their release: a pull request on MaDoc.

The production models are the ones on MaDoc's branch (models/blla.mlmodel, models/htr.mlmodel), described by
models/MODEL_CARD.json. The worker keeps a clone of MaDoc in the release volume, used only to read them and to
push the branch of a pull request; the deployed app is changed only when a person merges it.

GitHub: GH_TOKEN (.env), a fine-grained token with "Contents: read and write" and "Pull requests: read and write"
on MaDoc only. It is passed to git as a header for each command, never written in the clone.
"""

import base64
import json
import os
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone

from mlops.config import params, path


def clone_dir():
    return path("release") / "madoc"


def git_env():
    token = os.environ.get("GH_TOKEN", "")
    if not token:
        return {}
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}", "GIT_TERMINAL_PROMPT": "0"}


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd or clone_dir(), check=True, text=True, capture_output=True,
                          env={**os.environ, **git_env()}).stdout.strip()


def remote_url():
    return os.environ.get("MLOPS_MADOC_REMOTE") or f'https://github.com/{params()["release"]["github_repo"]}.git'


def sync():
    """The clone on the latest commit of MaDoc's branch (made the first time) -> its folder."""
    folder, branch = clone_dir(), params()["release"]["branch"]
    if not (folder / ".git").is_dir():
        shutil.rmtree(folder, ignore_errors=True)
        git("clone", "--branch", branch, remote_url(), str(folder), cwd=folder.parent)
    git("fetch", "origin", branch)
    git("checkout", "-B", branch, f"origin/{branch}")
    git("reset", "--hard", f"origin/{branch}")
    git("clean", "-fdx")
    return folder


def card():
    file = clone_dir() / params()["release"]["card"]
    return json.loads(file.read_text(encoding="utf-8")) if file.is_file() else {}


def production(task):
    """{"path": the production model of the task, "card": its entry of MODEL_CARD.json ({} when none)}."""
    return {"path": clone_dir() / params()["release"]["models"][task], "card": card().get(task, {})}


def wandb_alias(task, artifact, alias):
    """Best effort: an alias on a model version of W&B (W&B unreachable does not stop the job)."""
    try:
        import wandb
        settings = params()["wandb"]
        version = wandb.Api().artifact(f'{settings["entity"]}/{settings["project"][task]}/{artifact}')
        if alias not in version.aliases:
            version.aliases.append(alias)
            version.save()
    except Exception as e:
        print(f"W&B alias {alias} not set on {artifact}: {e}")


def mark_production():
    """W&B alias "production" on the versions MaDoc uses now (the registry follows the merged pull requests)."""
    for task, entry in card().items():
        if isinstance(entry, dict) and entry.get("artifact"):
            wandb_alias(task, entry["artifact"], "production")


def comparison_markdown(evaluation):
    rows = ["| Metric | Production | Candidate |", "|---|---|---|"]
    production_metrics = evaluation["production"] or {}
    for key, value in sorted(evaluation["candidate"].items()):
        if isinstance(value, (int, float)):
            rows.append(f"| {key} | {production_metrics.get(key, '—')} | {value} |")
    c = evaluation["comparison"]
    rows += ["", f'**Gain on {c["metric"]}** (per test page, positive = better): {c["gain"]:+.3f}, '
                 f'{int(100 * c["confidence"])} % CI [{c["low"]:+.3f}, {c["high"]:+.3f}] over {c["pages"]} pages']
    for subset in ("old", "new"):
        if c.get(subset):
            rows.append(f'- {subset} test pages ({c[subset]["pages"]}): gain {c[subset]["gain"]:+.3f}')
    runs = c.get("wandb") or {}
    links = [f"[{role} run]({runs[role]})" for role in ("candidate", "production") if runs.get(role)]
    if links:
        rows.append("\nW&B: " + " · ".join(links))
    return "\n".join(rows)


def create_pull_request(title, body, head):
    """POST /repos/{repo}/pulls -> URL of the pull request."""
    settings = params()["release"]
    request = urllib.request.Request(
        f'https://api.github.com/repos/{settings["github_repo"]}/pulls',
        data=json.dumps({"title": title, "body": body, "head": head, "base": settings["branch"]}).encode(),
        headers={"Authorization": f'Bearer {os.environ.get("GH_TOKEN", "")}', "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28"}, method="POST")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())["html_url"]


def open_pull_request(task, model, evaluation, dataset_version, log=print):
    """Copy the candidate into the clone, update MODEL_CARD.json, push a branch, open the pull request -> URL."""
    settings = params()["release"]
    folder = sync()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    branch = f"models/{task}-{stamp}"
    git("checkout", "-b", branch)
    shutil.copyfile(model["path"], folder / settings["models"][task])
    current = card()
    current[task] = {"artifact": model["artifact"], "dataset_version": dataset_version["digest"],
                     "pages": dataset_version["counts"], "trained_at": model["created_at"].isoformat(),
                     "test_version": evaluation["test_digest"], "test_metrics": evaluation["candidate"],
                     "evaluation": evaluation["comparison"]}
    card_file = folder / settings["card"]
    card_file.write_text(json.dumps(current, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    git("add", settings["models"][task], settings["card"])
    title = f'models: {task} {model["artifact"]}'
    git("-c", "user.name=MaDoc MLOps", "-c", "user.email=mlops@madoc.local", "commit", "-m", title)
    git("push", "origin", branch)
    log(f"branch {branch} pushed")
    body = "\n\n".join([
        f"Trained and evaluated by the MLOps platform. Candidate and production evaluated on the same "
        f'{evaluation["comparison"]["pages"]} test pages (test version {evaluation["test_digest"]}).',
        comparison_markdown(evaluation),
        f'Gate: {evaluation["message"]}',
        "Merging deploys the model (MaDoc's deploy workflow). Revert this pull request to roll it back.",
    ])
    url = create_pull_request(title, body, branch)
    wandb_alias(task, model["artifact"], "staging")
    return url