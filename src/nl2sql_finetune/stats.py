"""The one resampling routine this study uses, so both analyses share it.

A percentile bootstrap over **questions**. Questions are the independent unit: the ten
attempts inside a question are repeats of the same draw, so resampling attempts instead
would shrink every interval and manufacture confidence that was never measured.

Paired by construction -- the caller passes one value per question that is already a
difference between the two arms, so each resample takes both arms' results for the same
drawn questions.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Hashable

RESAMPLES = 10_000


def bootstrap(per_question: dict[Hashable, float], questions: list, *, seed: int = 0,
              resamples: int = RESAMPLES) -> tuple[float, float]:
    """95% percentile interval for the mean of a paired per-question difference."""
    rng = random.Random(seed)
    n = len(questions)
    values = [per_question[q] for q in questions]
    means = []
    for _ in range(resamples):
        means.append(statistics.fmean(values[rng.randrange(n)] for _ in range(n)))
    means.sort()
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]
