"""Test settings: a PostgreSQL database (MLOPS_TEST_DATABASE_URL, emptied for every test), temporary folders for
the data, the runs and the release clone, W&B offline, fake ketos / kraken (tests/fakes) on the PATH."""

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest
from PIL import Image

FAKES = Path(__file__).parent / "fakes"
LINE = ('<TextLine id="l{n}"><Coords points="0,{y} 100,{y} 100,{y2} 0,{y2}"/>'
        '<Baseline points="0,{y2} 100,{y2}"/><TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>')
WORDS = ["رسم", "ولادة", "رقم", "في", "يوم", "عام", "موافق", "سنة", "جماعة", "خريبڭة"]


def page_xml(image_name, lines=6, texts=None, baselines=True, width=200, height=300):
    body = []
    for n in range(lines):
        text = (texts or [" ".join(WORDS[(n + k) % len(WORDS)] for k in range(4)) for _ in range(lines)])[n]
        line = LINE.format(n=n, y=10 * n, y2=10 * n + 8, text=text)
        if not baselines:
            line = line.replace(f'<Baseline points="0,{10 * n + 8} 100,{10 * n + 8}"/>', "")
        body.append(line)
    return ('<?xml version="1.0" encoding="UTF-8"?><PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/'
            f'pagecontent/2019-07-15"><Page imageFilename="{image_name}" imageWidth="{width}" '
            f'imageHeight="{height}"><TextRegion id="r1">{"".join(body)}</TextRegion></Page></PcGts>')


def make_page(folder, stem, color, **kwargs):
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 300), color).save(folder / f"{stem}.jpg")
    (folder / f"{stem}.xml").write_text(page_xml(f"{stem}.jpg", **kwargs), encoding="utf-8")


def make_zip(folder, target):
    with zipfile.ZipFile(target, "w") as archive:
        for file in sorted(folder.rglob("*")):
            if file.is_file():
                archive.write(file, f"export/{file.relative_to(folder)}")
    return target


@pytest.fixture
def env(tmp_path, monkeypatch):
    url = os.environ.get("MLOPS_TEST_DATABASE_URL")
    if not url:
        pytest.skip("MLOPS_TEST_DATABASE_URL is not set (a PostgreSQL database the tests may empty)")
    import psycopg
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
    for name in ("data", "runs", "release"):
        monkeypatch.setenv(f"MLOPS_{name.upper()}_DIR", str(tmp_path / name))
    monkeypatch.setenv("MLOPS_DATABASE_URL", url)
    monkeypatch.setenv("MLOPS_SECRET", "test-secret")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    monkeypatch.setenv("PATH", f"{FAKES}{os.pathsep}{os.environ['PATH']}")
    from mlops import db
    db.init_schema()
    return tmp_path


@pytest.fixture
def madoc_remote(env, monkeypatch):
    """A bare git repository standing for MaDoc, with production models in models/."""
    bare = env / "madoc.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "master", str(bare)], check=True)
    seed = env / "seed"
    (seed / "models").mkdir(parents=True)
    (seed / "models" / "blla.mlmodel").write_text("production blla")
    (seed / "models" / "htr.mlmodel").write_text("production htr")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", "-b", "master", str(seed)], check=True)
    subprocess.run([*git, "add", "-A"], cwd=seed, check=True)
    subprocess.run([*git, "commit", "-qm", "init"], cwd=seed, check=True)
    subprocess.run(["git", "push", "-q", str(bare), "master"], cwd=seed, check=True)
    shutil.rmtree(seed)
    monkeypatch.setenv("MLOPS_MADOC_REMOTE", str(bare))
    return bare
