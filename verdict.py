# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""The one-line verdict over a result (green, amber or red) and the word for its Tidiness score. No bpy: unit tested."""

from __future__ import annotations

# First guesses, to tune on the bench set (DESIGN.md 2.8). Every threshold of the verdict lives here.
SERIOUS = ("inward", "folded_over", "crossing")  # bad-face kinds that break shading or a bake
GOOD_CHANGE = 1.2  # view change up to this many times the Detail asked is still "good"
CHECK_CHANGE = 2.0  # past this it is a real problem
CHECK_PER_1000 = 5.0  # serious bad faces per 1,000 faces: below is "check", above is a real problem
TIDY_BANDS = ((30.0, "rough"), (50.0, "machine-made"), (70.0, "like a quick hand retopo"), (90.0, "clean"))
TIDY_TOP = "hand-made rows"


def tidiness_words(score: float) -> str:
    for limit, words in TIDY_BANDS:
        if score < limit:
            return words
    return TIDY_TOP


def serious_count(totals: dict | None) -> int:
    return int(sum(int((totals or {}).get(k, 0)) for k in SERIOUS))


def verdict(use: str, faces: int, loss: float, asked: float | None, serious: int, spots: int) -> tuple[str, str]:
    """(level, title): level is GOOD, CHECK or BAD. asked is the Detail asked for (None when a face count was asked);
    serious the serious bad faces, spots the places they sit in."""
    ratio = loss / asked if asked else 0.0
    rate = 1000.0 * serious / max(faces, 1)
    where = f"{spots} spot{'s' if spots != 1 else ''}"
    if rate > CHECK_PER_1000:
        return "BAD", f"{where} with bad faces left"
    if ratio > CHECK_CHANGE:
        return "BAD", "Changed far more than asked"
    if serious:
        return "CHECK", f"Check {where}"
    if ratio > GOOD_CHANGE:
        return "CHECK", "Changed a little more than asked"
    return "GOOD", f"Good for {use.lower()}" if use else "Good"


if __name__ == "__main__":
    assert verdict("Animation", 6000, 1.5, 1.5, 0, 0) == ("GOOD", "Good for animation")
    assert verdict("Game Asset", 1000, 1.0, 6.0, 2, 1)[0] == "CHECK"
    assert verdict("Game Asset", 1000, 1.0, 6.0, 20, 4)[0] == "BAD"
    assert tidiness_words(56) == "like a quick hand retopo"
    print("ok")
