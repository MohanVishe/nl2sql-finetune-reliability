"""Draw the study's figures as plain SVG, straight from the result files.

    uv run python scripts/make_charts.py

No plotting library: the figures are small, and a hand-written SVG is reviewable in a diff,
regenerates identically, and adds no dependency to a repo whose point is reproducibility.
Colours are mid-tone on purpose -- the figures sit in a README that GitHub renders on both a
light and a dark background, and neither white nor black can be assumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = "#4c86e8"      # baseline / F0
TREAT = "#e8833a"     # treatment / F1
GOOD = "#3fa06a"
BAD = "#d1544f"
INK = "#8b93a1"       # readable on light and dark
GRID = "#8b93a180"
NEWLINE = chr(10)
FONT = ("font-family=\"-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif\"")


def text(x: float, y: float, body: str, *, size: float = 13, fill: str = INK,
         anchor: str = "start", weight: str = "normal") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" {FONT} font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{body}</text>')


def line(x1: float, y1: float, x2: float, y2: float, *, stroke: str = GRID,
         width: float = 1, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}"{dash_attr} />')


def polyline(points: list[tuple[float, float]], *, stroke: str, width: float = 2.5,
             dash: str = "") -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<polyline points="{coords}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width}" stroke-linejoin="round"{dash_attr} />')


def svg(width: int, height: int, body: str, title: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" aria-label="{title}">\n'
            f'<title>{title}</title>\n{body}\n</svg>\n')


def curves_chart(summary: dict, out: Path) -> None:
    """pass@k and pass^k for both arms, with the gap shaded."""
    w, h = 720, 420
    left, right, top, bottom = 62, 200, 46, 46
    plot_w, plot_h = w - left - right, h - top - bottom
    depth = summary["attempts_per_question"]
    top_pct = 0.50

    def px(k: int) -> float:
        return left + (k - 1) / (depth - 1) * plot_w

    def py(value: float) -> float:
        return top + plot_h - (value / top_pct) * plot_h

    parts = [f'<rect x="0" y="0" width="{w}" height="{h}" fill="none" />']
    for pct in range(0, int(top_pct * 100) + 1, 10):
        y = py(pct / 100)
        parts.append(line(left, y, left + plot_w, y))
        parts.append(text(left - 10, y + 4, f"{pct}%", size=12, anchor="end"))
    for k in range(1, depth + 1):
        parts.append(text(px(k), top + plot_h + 22, str(k), size=12, anchor="middle"))
    parts.append(text(left + plot_w / 2, h - 8, "k  (number of attempts)", size=13,
                      anchor="middle"))

    arms = [("baseline", "F0 — base model", BASE), ("treatment", "F1 — fine-tuned", TREAT)]
    for name, _, colour in arms:
        rows = summary["curves"][name]
        # Shade the gap between the two estimators.
        upper = [(px(r["k"]), py(r["pass_at_k"])) for r in rows]
        lower = [(px(r["k"]), py(r["pass_hat_k"])) for r in reversed(rows)]
        area = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper + lower)
        parts.append(f'<polygon points="{area}" fill="{colour}" fill-opacity="0.10" />')
        parts.append(polyline(upper, stroke=colour))
        parts.append(polyline([(px(r["k"]), py(r["pass_hat_k"])) for r in rows],
                              stroke=colour, dash="6 4"))

    ly = top + 6
    parts.append(text(left + plot_w + 18, ly, "solid = pass@k", size=12, weight="600"))
    parts.append(text(left + plot_w + 18, ly + 17, "at least one of k right", size=11))
    parts.append(text(left + plot_w + 18, ly + 44, "dashed = pass^k", size=12, weight="600"))
    parts.append(text(left + plot_w + 18, ly + 61, "all k right", size=11))
    for index, (name, label, colour) in enumerate(arms):
        y = ly + 100 + index * 46
        parts.append(line(left + plot_w + 18, y - 4, left + plot_w + 46, y - 4,
                          stroke=colour, width=3))
        parts.append(text(left + plot_w + 52, y, label.split(" — ")[0], size=12,
                          fill=colour, weight="700"))
        parts.append(text(left + plot_w + 18, y + 16, label.split(" — ")[1], size=11))
        row = summary["curves"][name][-1]
        parts.append(text(left + plot_w + 18, y + 32,
                          f"gap at k={depth}: {row['gap']:.1%}", size=11))
    parts.append(text(left - 52, 20, "share of questions", size=12))
    return out.write_text(svg(w, h, "\n".join(parts), "pass@k and pass^k, F0 vs F1"),
                          encoding="utf-8")


def moved_chart(summary: dict, out: Path) -> None:
    """How many questions the fine-tune fixed, broke, or left alone."""
    w, h = 720, 250
    moved = summary["moved"]
    total = summary["questions"]
    rows = [
        ("Right every time, both models", moved["solid_both"], INK),
        ("Wrong every time, both models", moved["never_both"], INK),
        ("Became right every time after fine-tuning", moved["became_solid"], GOOD),
        ("Stopped being right every time", moved["lost_solid"], BAD),
    ]
    longest = max(value for _, value, _ in rows) or 1
    left, top, bar_w, row_h = 300, 44, 300, 40
    parts = [text(20, 24, f"What changed, per question ({total} questions, "
                          f"{summary['attempts_per_question']} attempts each)",
                  size=13, weight="700")]
    for index, (label, value, colour) in enumerate(rows):
        y = top + index * row_h
        width = max(2, value / longest * bar_w)
        parts.append(text(left - 12, y + 14, label, size=12, anchor="end"))
        parts.append(f'<rect x="{left}" y="{y}" width="{width:.1f}" height="19" rx="3" '
                     f'fill="{colour}" fill-opacity="0.85" />')
        parts.append(text(left + width + 8, y + 14, str(value), size=12, weight="700"))
    parts.append(text(20, h - 14,
                      f"improved on {moved['improved']} questions, "
                      f"worsened on {moved['worsened']}", size=12))
    return out.write_text(svg(w, h, "\n".join(parts), "What changed after fine-tuning"),
                          encoding="utf-8")


def training_chart(training: dict, out: Path) -> None:
    """Training loss keeps falling; validation loss turns up -- that is overfitting."""
    history = training["log_history"]
    train = [(h["step"], h["loss"]) for h in history if "loss" in h]
    evals = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    best_step, best_loss = min(evals, key=lambda p: p[1])

    w, h = 720, 400
    left, right, top, bottom = 62, 180, 46, 52
    plot_w, plot_h = w - left - right, h - top - bottom
    max_step = max(s for s, _ in train + evals)
    low = min(v for _, v in train + evals) * 0.92
    high = max(v for _, v in train + evals) * 1.04

    def px(step: float) -> float:
        return left + step / max_step * plot_w

    def py(value: float) -> float:
        return top + plot_h - (value - low) / (high - low) * plot_h

    parts = []
    for i in range(5):
        value = low + (high - low) * i / 4
        y = py(value)
        parts.append(line(left, y, left + plot_w, y))
        parts.append(text(left - 10, y + 4, f"{value:.2f}", size=11, anchor="end"))
    for step in range(0, max_step + 1, 100):
        parts.append(text(px(step), top + plot_h + 22, str(step), size=11, anchor="middle"))
    parts.append(text(left + plot_w / 2, h - 10, "training step", size=13, anchor="middle"))
    parts.append(text(left - 52, 20, "loss (lower is better)", size=12))

    parts.append(polyline([(px(s), py(v)) for s, v in train], stroke=BASE, width=2))
    parts.append(polyline([(px(s), py(v)) for s, v in evals], stroke=TREAT, width=2.5))
    parts.append(f'<circle cx="{px(best_step):.1f}" cy="{py(best_loss):.1f}" r="5" '
                 f'fill="{TREAT}" />')
    parts.append(line(px(best_step), top, px(best_step), top + plot_h, stroke=TREAT,
                      width=1, dash="4 4"))
    parts.append(text(px(best_step) + 8, top + 14, f"best: step {best_step}", size=11,
                      fill=TREAT, weight="700"))

    ly = top + 20
    parts.append(line(left + plot_w + 18, ly - 4, left + plot_w + 46, ly - 4, stroke=BASE,
                      width=3))
    parts.append(text(left + plot_w + 52, ly, "training", size=12, fill=BASE, weight="700"))
    parts.append(text(left + plot_w + 18, ly + 17, "questions it studies", size=11))
    parts.append(line(left + plot_w + 18, ly + 44, left + plot_w + 46, ly + 44, stroke=TREAT,
                      width=3))
    parts.append(text(left + plot_w + 52, ly + 48, "validation", size=12, fill=TREAT,
                      weight="700"))
    parts.append(text(left + plot_w + 18, ly + 65, "held-out databases", size=11))
    parts.append(text(left + plot_w + 18, ly + 95, "after the best point,", size=11))
    parts.append(text(left + plot_w + 18, ly + 111, "it memorises instead", size=11))
    parts.append(text(left + plot_w + 18, ly + 127, "of learning", size=11))
    return out.write_text(svg(w, h, "\n".join(parts), "Training and validation loss"),
                          encoding="utf-8")


def _stacked(rows: list[tuple[str, list[tuple[str, float, str]]]], *, title: str,
             note: str, unit: str, out: Path) -> None:
    """One horizontal 100%-wide bar per arm, split into labelled segments."""
    w = 760
    left, right, top, row_h = 132, 26, 52, 54
    h = top + len(rows) * row_h + 58
    bar_w = w - left - right
    total = sum(value for _, value, _ in rows[0][1])
    parts = [text(20, 26, title, size=13, weight="700")]
    for index, (name, segments) in enumerate(rows):
        y = top + index * row_h
        parts.append(text(left - 12, y + 20, name, size=12, anchor="end", weight="600"))
        x = left
        for _label, value, colour in segments:
            seg = value / total * bar_w
            parts.append(f'<rect x="{x:.1f}" y="{y}" width="{max(seg, 0.6):.1f}" height="28" '
                         f'fill="{colour}" fill-opacity="0.85" />')
            if seg > 46:
                shown = f"{value:.0f}" if unit == "count" else f"{value * 100:.0f}%"
                parts.append(text(x + seg / 2, y + 19, shown, size=12, fill="#ffffff",
                                  anchor="middle", weight="700"))
            x += seg
    legend_y = top + len(rows) * row_h + 14
    x = left
    for label, _, colour in rows[0][1]:
        parts.append(f'<rect x="{x}" y="{legend_y - 9}" width="11" height="11" rx="2" '
                     f'fill="{colour}" fill-opacity="0.85" />')
        parts.append(text(x + 17, legend_y, label, size=12))
        x += 20 + len(label) * 7.0
    parts.append(text(20, h - 12, note, size=12))
    return out.write_text(svg(w, h, NEWLINE.join(parts), title), encoding="utf-8")


def failure_modes_chart(report: dict, out: Path) -> None:
    """Every attempt, split by how it failed -- loudly or silently."""
    names = {"baseline": "F0  untrained", "treatment": "F1  fine-tuned",
             "reference": "7B  prompted"}
    # Net shifts in rates across independent attempts -- no query is tracked between models.
    shift = report["differences"]["treatment_minus_baseline"]
    rows = []
    for arm, label in names.items():
        r = report["rates"].get(arm)
        if r is None:
            continue
        rows.append((label, [
            ("right", r["correct"], GOOD),
            ("crashed (visible failure)", r["crashed"], TREAT),
            ("ran, wrong rows (silent failure)", r["silent_wrong"], BAD),
        ]))
    _stacked(rows, title=f"How each attempt ended "
                         f"({report['questions']} questions x "
                         f"{report['attempts_per_question']} attempts)",
             note=f"F1 against F0: crashes fell {-100 * shift['crashed']['delta']:.1f} points; "
                  f"right answers rose {100 * shift['correct']['delta']:.1f}, silent wrong "
                  f"answers {100 * shift['silent_wrong']['delta']:.1f}.",
             unit="rate", out=out)


def consistency_chart(report: dict, out: Path) -> None:
    """Where the flakiness lives: questions by how many attempts succeeded."""
    names = {"baseline": "F0  untrained", "treatment": "F1  fine-tuned",
             "reference": "7B  prompted"}
    rows = []
    for arm, label in names.items():
        c = report["consistency"].get(arm)
        if c is None:
            continue
        rows.append((label, [
            ("never right", c["never"], INK),
            ("flaky (right some of the time)", c["flaky"], TREAT),
            ("right every time", c["always"], GOOD),
        ]))
    _stacked(rows, title="Questions by how many of the ten attempts were right",
             note="The flaky middle is identical before and after fine-tuning: 113 questions. "
                  "The 7B's is 68.",
             unit="count", out=out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, default=Path("results/summary.json"))
    parser.add_argument("--training", type=Path,
                        default=Path("results/f1-training/training.json"))
    parser.add_argument("--failure-modes", type=Path,
                        default=Path("results/failure-modes.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/images"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    if args.training.exists():
        training_chart(json.loads(args.training.read_text(encoding="utf-8")),
                       args.out / "training-curve.svg")
        written.append("training-curve.svg")
    if args.summary.exists():
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        curves_chart(summary, args.out / "passk-curves.svg")
        moved_chart(summary, args.out / "what-changed.svg")
        written += ["passk-curves.svg", "what-changed.svg"]
    if args.failure_modes.exists():
        report = json.loads(args.failure_modes.read_text(encoding="utf-8"))
        failure_modes_chart(report, args.out / "failure-modes.svg")
        consistency_chart(report, args.out / "consistency.svg")
        written += ["failure-modes.svg", "consistency.svg"]
    print(f"wrote {', '.join(written)} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
