"""Settings changed in the app, on top of params.yaml.

params.yaml gives the defaults; what is chosen in the app (Train tab: the training options of each task; Data tab:
the split) is saved in the database (table settings) and used from then on, until it is changed again or reset.
A training job also keeps the exact options it ran with (its args, its W&B config): changing the settings later
never changes what an old run did.
"""

import shlex

from mlops import db
from mlops.config import params

# the training options of the Train tab: name -> (type, help). None (empty in the app) = kraken's default.
TRAIN_FIELDS = {
    "start_from": (str, "model of the MaDoc repository to fine-tune (its production model by default)"),
    "epochs": (int, "-1: until early stopping"),
    "quit": (str, "early: stop after `lag` epochs without a better validation; fixed: run `epochs` epochs"),
    "lag": (int, "epochs without a better validation before early stopping"),
    "min_epochs": (int, "epochs always trained before early stopping may stop"),
    "batch_size": (int, "lines per batch (HTR only)"),
    "resize": (str, "union: keep the model's alphabet and add the new characters; new: exactly the data's"),
    "learning_rate": (float, "empty: kraken's default (HTR 1e-3, blla 2e-4)"),
    "schedule": (str, "constant, or decreasing: cosine (over `cos max` epochs), exponential, step, reduceonplateau"),
    "cos_max": (int, "cosine schedule: epochs over which the learning rate decreases"),
    "kraken_augment": (bool, "kraken's own augmentation of the lines while training (ketos --augment)"),
    "use_augmented": (bool, "add the augmented train pages of the dataset version (Data tab)"),
    "extra_args": (str, "any other ketos option, as on the command line"),
}
TASK_FIELDS = {"htr": list(TRAIN_FIELDS), "blla": [k for k in TRAIN_FIELDS if k != "batch_size"]}
CHOICES = {"quit": ["early", "fixed"], "resize": ["union", "new"],
           "schedule": ["constant", "cosine", "exponential", "step", "reduceonplateau", "1cycle"]}
SPLIT_KEYS = ("val_fraction", "test_fraction", "split_tolerance", "split_weights")


def saved(conn, name):
    row = db.one(conn, "SELECT value FROM settings WHERE name = %s", name)
    return row["value"] if row else {}


def save(conn, name, value):
    conn.execute("INSERT INTO settings (name, value) VALUES (%s, %s) ON CONFLICT (name) DO UPDATE "
                 "SET value = EXCLUDED.value, updated_at = now()", (name, db.json(value)))


def reset(conn, name):
    conn.execute("DELETE FROM settings WHERE name = %s", (name,))


def default_training(task):
    """The defaults of params.yaml (older keys accepted: extra_args as a list)."""
    p = dict(params()[task])
    extra = p.get("extra_args") or []
    out = {"start_from": p.get("start_from") or params()["release"]["models"][task], "epochs": p.get("epochs", -1),
           "quit": p.get("quit", "early"), "lag": p.get("lag", 10), "min_epochs": p.get("min_epochs"),
           "batch_size": p.get("batch_size"), "resize": p.get("resize", "union"),
           "learning_rate": p.get("learning_rate"), "schedule": p.get("schedule", "constant"),
           "cos_max": p.get("cos_max"), "kraken_augment": bool(p.get("kraken_augment", False)),
           "use_augmented": bool(p.get("use_augmented", False)),
           "extra_args": extra if isinstance(extra, str) else shlex.join(str(a) for a in extra)}
    return {k: v for k, v in out.items() if k in TASK_FIELDS[task]}


# the smallest value of the numeric options (None: kraken's default); epochs -1 is "until early stopping"
MINIMUM = {"epochs": -1, "lag": 1, "min_epochs": 0, "batch_size": 1, "cos_max": 1}


def typed(key, value):
    """One option from the app -> its type (empty -> None), or a ValueError that names the option."""
    kind = TRAIN_FIELDS[key][0]
    if value is None or (isinstance(value, str) and not value.strip() and kind is not str):
        return None
    if kind is bool:
        return bool(value)
    if kind is str:
        return str(value).strip()
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key}: {value!r} is not a number") from None
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{key}: {value!r} is not a finite number")
    if kind is int:
        if number != int(number):
            raise ValueError(f"{key}: {value!r} is not a whole number")
        number = int(number)
    if key in MINIMUM and number < MINIMUM[key]:
        raise ValueError(f"{key}: {number} is below {MINIMUM[key]}")
    if key == "learning_rate" and number <= 0:
        raise ValueError(f"{key}: {number} must be above 0")
    return number


def clean(task, values):
    """Values from the app -> typed values (empty -> None), only the fields of the task."""
    out = {key: typed(key, values.get(key)) for key in TASK_FIELDS[task]}
    for key in CHOICES:
        if key in out and out[key] not in CHOICES[key] + [None]:
            raise ValueError(f"{key}: {out[key]!r} is not one of {CHOICES[key]}")
    if out.get("epochs") == 0:
        raise ValueError("epochs: 0 trains nothing (-1: until early stopping)")
    if out.get("quit") == "fixed" and (out.get("epochs") is None or out["epochs"] < 1):
        raise ValueError("quit fixed runs `epochs` epochs: set epochs to 1 or more (with -1 it would never stop)")
    try:
        shlex.split(out.get("extra_args") or "")   # a quote left open raises here, not in the job
    except ValueError as e:
        raise ValueError(f"extra_args: {e}") from None
    return out


def training(conn, task):
    """The training options now: params.yaml, then what was saved from the app."""
    return {**default_training(task), **{k: v for k, v in saved(conn, f"train.{task}").items()
                                         if k in TASK_FIELDS[task]}}


def default_split():
    ingest = params()["ingest"]
    return {"val_fraction": ingest["val_fraction"], "test_fraction": ingest["test_fraction"],
            "split_tolerance": ingest.get("split_tolerance", 0.33),
            "split_weights": ingest.get("split_weights") or {}}


def split(conn):
    """The split settings now: params.yaml (ingest), then what was saved from the app."""
    out = default_split()
    for key, value in saved(conn, "split").items():
        if key in SPLIT_KEYS:
            out[key] = value
    return out


def check_split(value):
    val, test, tolerance = value["val_fraction"], value["test_fraction"], value["split_tolerance"]
    if not (0 < val < 1 and 0 < test < 1 and val + test < 1):
        raise ValueError("val and test must be between 0 and 100 %, together below 100 %")
    if not 0 <= tolerance <= 1:
        raise ValueError("the tolerance must be between 0 and 100 %")
    weights = value["split_weights"]
    if any(float(w) < 0 for w in [weights.get("missing_character", 0), *(weights.get("page_types") or {}).values()]):
        raise ValueError("the weights cannot be negative")
    return value


def publish_models(conn, folder):
    """The .mlmodel files of the MaDoc clone (the worker has it; the app reads the list from here)."""
    names = sorted(str(p.relative_to(folder)) for p in folder.rglob("*.mlmodel") if ".git" not in p.parts)
    save(conn, "madoc_models", {"models": names})
    return names


def madoc_models(conn):
    return saved(conn, "madoc_models").get("models") or []


def paused(conn):
    """The worker takes no new job while paused (a deployment waits for the running job, then restarts it)."""
    return bool(saved(conn, "worker").get("paused"))


def pause(conn, reason="deployment"):
    save(conn, "worker", {"paused": True, "reason": reason})


def resume(conn):
    reset(conn, "worker")