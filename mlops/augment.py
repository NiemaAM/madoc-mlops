"""Augmentation of the train pages of a dataset version (script/data_augmentation_xml.py).

Only the train pages: the validation and test pages stay as they are. The appearance changes (noise, lighting,
shadow, blur, scribbles) and the geometry changes (rotation, perspective) move the lines of the PageXML with the
image, so the augmented pages serve the HTR and the segmentation (blla).

Started by an "augment" job (Data tab: copies, seed) or by a training that uses augmented pages and finds none for
its version. The pages are kept in <data>/augmented/<version>-c<copies>-s<seed>/ and reused by every training of
that version; the augmentations of older versions are deleted (disk).
"""

import shutil
import sys

from mlops import datasets
from mlops.config import ROOT, params, path


def folder_name(version, copies, seed):
    return f'{version["digest"]}-c{copies}-s{seed}'


def augmentations():
    """The finished augmentations, the newest first -> [{"folder", "digest", "copies", "seed", "pages"}]."""
    root = path("data") / "augmented"
    out = []
    for folder in root.iterdir() if root.is_dir() else []:
        if not (folder / ".done").is_file():
            continue
        digest, _, rest = folder.name.partition("-c")
        copies, _, seed = rest.partition("-s")
        out.append({"folder": folder, "digest": digest, "copies": copies, "seed": seed,
                    "pages": sum(1 for _ in folder.glob("*.xml")), "time": (folder / ".done").stat().st_mtime})
    return sorted(out, key=lambda a: -a["time"])


def latest(version):
    """The newest finished augmentation of this dataset version, or None."""
    return next((a for a in augmentations() if a["digest"] == version["digest"]), None)


def augment(version, train_rows, log, should_stop=lambda: False, copies=None, seed=None):
    """-> the XML paths of the augmented pages of the version (made if needed)."""
    from mlops.process import run

    settings = params()["augment"]
    copies = int(copies or settings["copies"])
    seed = int(settings["seed"] if seed is None else seed)
    target = path("data") / "augmented" / folder_name(version, copies, seed)
    done = target / ".done"
    if not done.is_file():
        log(f"augmenting {len(train_rows)} train pages of version {version['digest']} ({copies} copies, seed {seed})")
        source = path("data") / "work" / f'augment-{version["digest"]}'
        shutil.rmtree(source, ignore_errors=True)
        source.mkdir(parents=True)
        pages = path("data") / "pages"
        for row in train_rows:   # a folder with only the train pages: links, not copies
            for name in (row["image_name"], f'{row["id"]}.xml'):
                (source / name).symlink_to(pages / name)
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        run([sys.executable, ROOT / "script" / "data_augmentation_xml.py", "--input", source, "--output", target,
             "--copies", copies, "--seed", seed], log, should_stop)
        done.write_text("ok\n")
        shutil.rmtree(source, ignore_errors=True)
        for old in (path("data") / "augmented").iterdir():   # the augmentations of older versions
            if old.is_dir() and not old.name.startswith(version["digest"]):
                shutil.rmtree(old)
    else:
        log(f"augmented pages of version {version['digest']} reused ({copies} copies, seed {seed})")
    return sorted(str(p) for p in target.glob("*.xml"))


def augmented_paths(version, train_rows, log, should_stop):
    """For a training: the newest augmentation of the version, else one made with params.yaml (augment)."""
    found = latest(version)
    if found:
        log(f"augmented pages: {found['folder'].name} ({found['pages']} pages)")
        return sorted(str(p) for p in found["folder"].glob("*.xml"))
    return augment(version, train_rows, log, should_stop)


def run_augment(conn, job, log, should_stop, set_wandb_url):
    """The "augment" job: the train pages of the current dataset version."""
    version = datasets.snapshot(conn)
    rows = datasets.pages(conn, version["id"], "train")
    if not rows:
        raise RuntimeError("no train pages")
    args = job["args"] or {}
    made = augment(version, rows, log, should_stop, args.get("copies"), args.get("seed"))
    log(f"{len(made)} augmented pages for version {version['digest']}")
    return {"dataset_version": version["digest"], "pages": len(made)}