"""Augmentation of the train pages of a dataset version (script/data_augmentation_xml.py).

Only the train pages: the validation and test pages stay as they are. The appearance changes (noise, lighting,
shadow, blur, scribbles) and the geometry changes (rotation, perspective) move the lines of the PageXML with the
image, so the augmented pages serve the HTR and the segmentation (blla).

Started by an "augment" job (Data tab: copies, seed) or by a training that uses augmented pages and finds none for
its version. The pages are kept in <data>/augmented/<version>-c<copies>-s<seed>/ and reused by every training of
that version; the augmentations of older versions are deleted (disk).
"""

import os
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


def reuse(row, copies, seed, target):
    """Link the augmented copies of this page from a finished augmentation with the same copies and seed into
    `target` -> True when all of them were found."""
    for other in augmentations():
        if other["folder"] == target or other["copies"] != str(copies) or other["seed"] != str(seed):
            continue
        files = []
        for k in range(1, copies + 1):
            stem = f'{row["id"]}_aug{k}'
            xml = other["folder"] / f"{stem}.xml"
            images = [f for f in other["folder"].glob(f"{stem}.*") if f.suffix != ".xml"]
            if not xml.is_file() or not images:
                break
            files += [xml, images[0]]
        else:
            for file in files:
                destination = target / file.name
                try:
                    os.link(file, destination)       # no copy: the same file in both folders
                except OSError:
                    shutil.copyfile(file, destination)
            return True
    return False


def augment(version, train_rows, log, should_stop=lambda: False, copies=None, seed=None):
    """-> the XML paths of the augmented pages of the version (made if needed)."""
    from mlops.process import run

    settings = params()["augment"]
    copies = int(copies or settings["copies"])
    seed = int(settings["seed"] if seed is None else seed)
    target = path("data") / "augmented" / folder_name(version, copies, seed)
    done = target / ".done"
    if not done.is_file():
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        # a page already augmented for another version (same copies, same seed) is the same: the seed of a copy
        # depends on the page, the copy and the seed only. Its files are linked, only the new pages are augmented.
        missing = [row for row in train_rows if not reuse(row, copies, seed, target)]
        log(f"version {version['digest']}: {len(train_rows) - len(missing)} train pages already augmented "
            f"(reused), {len(missing)} to augment ({copies} copies, seed {seed})")
        if missing:
            source = path("data") / "work" / f'augment-{version["digest"]}'
            shutil.rmtree(source, ignore_errors=True)
            source.mkdir(parents=True)
            pages = path("data") / "pages"
            for row in missing:   # a folder with only these pages: links, not copies
                for name in (row["image_name"], f'{row["id"]}.xml'):
                    (source / name).symlink_to(pages / name)
            run([sys.executable, ROOT / "script" / "data_augmentation_xml.py", "--input", source, "--output",
                 target, "--copies", copies, "--seed", seed], log, should_stop)
            shutil.rmtree(source, ignore_errors=True)
        done.write_text("ok\n")
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