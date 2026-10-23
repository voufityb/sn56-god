"""Task-aware additions to the per-family recipe, decided from what the validator hands the trainer."""

import re
from dataclasses import dataclass

from .data import read_rows

# Retraining on all images recovers the 3+ holdout images, which is a large share of a small dataset.
RETRAIN_ALL_MAX_IMAGES = 20
PLATEAU_EVALS = 2
# Mirroring an image mirrors any text in it, so flips are only used when no caption hints at rendered text.
# A double quote also catches quoted strings and the JSON prompts used for Ideogram tasks.
TEXT_HINT = re.compile(
    r'"|\b(text|logo|logos|sign|signs|signage|label|labels|letter|letters|lettering|typography|font|word|words|title|'
    r"headline|slogan|banner|button|buttons|menu|ui|interface|screen|poster|reads|says|saying)\b",
    re.IGNORECASE,
)


@dataclass
class TaskProfile:
    kind: str
    images: int
    text_risk: bool


def profile(dataset, trigger_word=None):
    rows = read_rows(dataset)
    captions = [caption or "" for _, _, caption in rows]
    return TaskProfile(
        # Validators attach a trigger word to person/logo/social/design/product tasks and none to style tasks.
        kind="subject" if trigger_word else "style",
        images=len(rows),
        text_risk=any(TEXT_HINT.search(caption) for caption in captions),
    )


def extra_args(task):
    args = []
    if task.kind == "style" and not task.text_risk:
        args.append("--hflip")
    if task.images <= RETRAIN_ALL_MAX_IMAGES:
        args += ["--retrain-all", "--retrain-max-images", str(RETRAIN_ALL_MAX_IMAGES),
                 "--retrain-on-plateau", str(PLATEAU_EVALS)]
    return args
