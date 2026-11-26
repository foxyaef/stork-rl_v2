"""Summarize a live_browser_test.py CSV log.

Usage:  python loop_stats.py logs\\cast_v9.csv

Prints the control-loop speed (the main limit on distance) and a rough
distance estimate for every game found in the log. In the SWF the distance
depends only on the time survived, so the estimate is computed from that time;
always prefer the score shown on screen.

A game is taken to end when the measured neck angle stops changing (the
game-over screen is a still image) and a new game to start when it moves
again, so games are found even when a quick restart kept the same session_id.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys

STATIC_GAP_S = 1.0  # a still image for this long means the game is over
MIN_GAME_S = 5.0  # shorter segments are usually menu clicks, not games


def metres_after(seconds: float) -> int:
    xs = xx = 0.0
    bi = 1.0
    for _ in range(int(seconds * 30)):
        xx += xs
        xs = bi * 3.0
        bi += 0.002
    return math.floor(xx / 550.0 * 3.0)


def find_games(rows: list[dict[str, str]]) -> list[tuple[int, float]]:
    """Return (session_id, seconds survived) for each game in the log."""
    games: list[tuple[int, float]] = []
    session = start = last_change = previous = None

    def close() -> None:
        if start is not None and last_change is not None and last_change - start >= MIN_GAME_S:
            games.append((session, last_change - start))

    for row in rows:
        if row["measurement_accepted"] != "True" or not row["body_measurement"]:
            continue
        now = float(row["elapsed_s"])
        measurement = float(row["body_measurement"])
        if int(row["session_id"]) != session:
            close()
            session = int(row["session_id"])
            start = now - float(row["session_elapsed_s"])
            last_change = previous = None
        if previous is not None and abs(measurement - previous) > 0.01 and -110.0 <= measurement <= 80.0:
            if last_change is not None and now - last_change >= STATIC_GAP_S:
                close()  # still screen, then motion again: a new game started
                start = now
            last_change = now
        previous = measurement
    close()
    return games


def main(path: str) -> None:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        print("Not enough rows.")
        return

    times = [float(row["elapsed_s"]) for row in rows]
    gaps = sorted((b - a) * 1000.0 for a, b in zip(times, times[1:]) if b > a)
    median_gap = statistics.median(gaps)
    print(
        f"loops={len(rows)}  median loop={median_gap:.0f} ms "
        f"(= {median_gap / 33.3:.1f} Flash frames)  "
        f"p90={gaps[int(len(gaps) * 0.9)]:.0f} ms"
    )
    captures = [float(row["capture_ms"]) for row in rows if row["capture_ms"]]
    print(f"median capture={statistics.median(captures):.0f} ms")
    holds = [float(row["input_ms"]) for row in rows if row["input_ms"] and float(row["input_ms"]) > 0]
    if holds:
        print(
            f"key presses={len(holds)}  median hold={statistics.median(holds):.0f} ms "
            f"(= {statistics.median(holds) / 33.3:.1f} frames)  max={max(holds):.0f} ms"
        )

    games = find_games(rows)
    for index, (session, seconds) in enumerate(games, start=1):
        print(
            f"game {index} (session {session}): survived about {seconds:.1f} s "
            f"-> roughly {metres_after(seconds)} m (estimate; record the on-screen score)"
        )
    if games:
        estimates = [metres_after(seconds) for _, seconds in games]
        print(f"games={len(games)}  mean estimate={statistics.mean(estimates):.1f} m")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python loop_stats.py <live csv>")
    main(sys.argv[1])
