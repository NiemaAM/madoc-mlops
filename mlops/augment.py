"""Augmentation of the train pages of a dataset version (script/data_augmentation_xml.py), made once per version.

Only the train pages: the validation and test pages stay as they are. The augmented pages of a version are kept in
<data>/augmented/<version>-c<copies>-s<seed>/ and reused by every training of that version.
"""

import shutil
import sys

from mlops.config import ROOT, params, path
from mlops.process import run


def augment(version, train_rows, log, should_stop=lambda: False):
    """-> the XML paths of the augmented pages (made if needed)."""
    settings = params()["augment"]
    target = path("data") / "augmented" / f'{version["digest"]}-c{settings["copies"]}-s{settings["seed"]}'
    done = target / ".done"
    if not done.is_file():
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
             "--copies", settings["copies"], "--seed", settings["seed"]], log, should_stop)
        done.write_text("ok\n")
        shutil.rmtree(source, ignore_errors=True)
        for old in (path("data") / "augmented").iterdir():   # the augmentations of older versions
            if old.is_dir() and old != target:
                shutil.rmtree(old)
    else:
        log(f"augmented pages of version {version['digest']} reused")
    return sorted(str(p) for p in target.glob("*.xml"))
