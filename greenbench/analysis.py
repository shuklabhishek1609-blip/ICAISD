"""Figures and tables for the paper.

Output is vector PDF for LaTeX plus PNG for quick viewing. These are print
figures, so they commit to the light surface only -- there is no dark variant,
deliberately.

Colour follows the validated reference palette. Two notes on why the charts
look the way they do:

  * The Pareto scatter carries identity in *text labels*, not hue. A scatter
    puts every pair of colours side by side, and no 7-hue categorical set
    survives that test; two roles (on-frontier / dominated) plus direct labels
    does survive, and reads better in print besides.
  * Aqua and yellow sit under 3:1 against the light surface, so every series
    that uses them is directly labelled. That is the documented relief for a
    contrast warning, not an oversight.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from .instrument import RunRecord, joules_to_kwh, carbon_grams, DEFAULT_GRID_INTENSITY

# --- palette (validated: see references/palette.md) ------------------------
SURFACE = '#fcfcfb'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
BASELINE = '#c3c2b7'

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']  # slots 1-4
ACCENT = SERIES[0]
DOMINATED = INK_MUTED

DISPLAY_NAME = {
    'hl': 'Historical Last',
    'nlinear': 'NLinear',
    'stid': 'STID',
    'lstm': 'LSTM',
    'stgcn': 'STGCN',
    'gwnet': 'Graph WaveNet',
    'sttn': 'STTN',
}


def load_results(results_dir='results', dataset=None):
    records = [RunRecord.load(p) for p in sorted(glob.glob(os.path.join(results_dir, '*.json')))]
    if dataset:
        records = [r for r in records if r.dataset == dataset]
    return records


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_SECONDARY)


def pareto_frontier(points):
    """Indices of non-dominated points, minimising both coordinates.

    A model is on the frontier when nothing else is both cheaper and more
    accurate. That is the whole argument of the paper in one function.
    """
    order = sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    frontier, best_y = [], float('inf')
    for i in order:
        if points[i][1] < best_y:
            frontier.append(i)
            best_y = points[i][1]
    return set(frontier)


def plot_pareto(records, cost='train_joules', out='figures/pareto', annotate_savings=True):
    """Accuracy against cost, with the non-dominated set called out.

    `cost` is any RunRecord numeric field -- train_joules for the training
    story, infer_joules_per_1k or latency_ms for the deployment story. Run it
    for both; they do not always agree, and where they disagree is interesting.
    """
    records = [r for r in records if np.isfinite(getattr(r, cost)) and np.isfinite(r.mae)]
    if not records:
        raise ValueError('no records with finite cost and MAE')

    xs = [getattr(r, cost) for r in records]
    ys = [r.mae for r in records]
    frontier = pareto_frontier(list(zip(xs, ys)))

    # Historical Last does no training at all, so its cost is a true zero --
    # and a zero has no position on a log axis, so matplotlib drops the point
    # without a word. That silently deletes the cheapest model from the figure
    # whose entire argument is about cheap models. Park zeros a decade below
    # the cheapest measurable model, draw them hollow, and label them with the
    # real value so the position reads as "off-scale zero" and not as data.
    positive = [x for x in xs if x > 0]
    floor = min(positive) / 10 if positive else 1.0
    plot_xs = [x if x > 0 else floor for x in xs]

    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    _style_axes(ax)

    # frontier line first, so markers sit on top
    fr = sorted(frontier, key=lambda i: xs[i])
    ax.plot([plot_xs[i] for i in fr], [ys[i] for i in fr],
            color=ACCENT, linewidth=2.0, alpha=0.45, zorder=2)

    for i, r in enumerate(records):
        on = i in frontier
        is_zero = xs[i] <= 0
        color = ACCENT if on else DOMINATED
        ax.plot(plot_xs[i], ys[i], marker='o', markersize=9 if on else 8,
                color=color,
                markerfacecolor=SURFACE if is_zero else color,
                markeredgecolor=color if is_zero else SURFACE, markeredgewidth=2.0,
                linestyle='none', zorder=3)
        name = DISPLAY_NAME.get(r.model, r.model)
        ax.annotate(f'{name} (0)' if is_zero else name, (plot_xs[i], ys[i]),
                    textcoords='offset points', xytext=(9, 4),
                    fontsize=9, color=INK_PRIMARY if on else INK_SECONDARY,
                    fontweight='bold' if on else 'normal', zorder=4)

    # room for the labels, which sit to the right of their markers
    ax.margins(x=0.16, y=0.12)

    ax.set_xscale('log')
    labels = {
        'train_joules': 'Training energy (J, log scale)',
        'infer_joules_per_1k': 'Inference energy per 1000 samples (J, log scale)',
        'latency_ms': 'Inference latency (ms, log scale)',
        'flops_per_sample': 'Forward FLOPs per sample (log scale)',
        'params': 'Parameters (log scale)',
    }
    ax.set_xlabel(labels.get(cost, cost), fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel('Test MAE (avg. over 12 horizons)', fontsize=10, color=INK_SECONDARY)

    # legend: two roles, always present
    handles = [
        plt.Line2D([], [], marker='o', markersize=9, color=ACCENT, linestyle='none',
                   markeredgecolor=SURFACE, markeredgewidth=2.0, label='Pareto-optimal'),
        plt.Line2D([], [], marker='o', markersize=8, color=DOMINATED, linestyle='none',
                   markeredgecolor=SURFACE, markeredgewidth=2.0, label='Dominated'),
    ]
    leg = ax.legend(handles=handles, frameon=False, fontsize=9, loc='upper right')
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)

    if annotate_savings and cost == 'train_joules':
        # Compare against the cheapest model that actually trains: a ratio
        # against Historical Last's zero is either a divide-by-zero or an
        # infinity, and neither is a sentence you can put in a figure title.
        paid = [i for i in frontier if xs[i] > 0]
        richest = max(range(len(records)), key=lambda i: xs[i])
        if paid and xs[richest] > 0:
            cheapest = min(paid, key=lambda i: xs[i])
            factor = xs[richest] / xs[cheapest]
            # Signed MAE deltas invite the wrong reading -- lower MAE is
            # better, so "+20.12 MAE" looks like a penalty when it is the
            # gain. Spell the direction out in words.
            gap = ys[cheapest] - ys[richest]
            direction = 'lower' if gap > 0 else 'higher'
            ax.set_title(
                f'{DISPLAY_NAME.get(records[richest].model, records[richest].model)} costs '
                f'{factor:,.0f}x the training energy of '
                f'{DISPLAY_NAME.get(records[cheapest].model, records[cheapest].model)} '
                f'for {abs(gap):.2f} {direction} MAE',
                fontsize=10, color=INK_PRIMARY, loc='left', pad=12)

    fig.tight_layout()
    _save(fig, out)
    return fig


def plot_horizon(records, models=None, out='figures/horizon'):
    """MAE against forecast step. Caps at four series -- see module docstring."""
    if models:
        records = [r for r in records if r.model in models]
    records = [r for r in records if r.horizon_mae]
    records = sorted(records, key=lambda r: r.mae)[:4]

    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=SURFACE)
    _style_axes(ax)

    ends = []
    for i, r in enumerate(records):
        steps = np.arange(1, len(r.horizon_mae) + 1)
        color = SERIES[i % len(SERIES)]
        ax.plot(steps, r.horizon_mae, color=color, linewidth=2.0,
                marker='o', markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=DISPLAY_NAME.get(r.model, r.model), zorder=3)
        ends.append((r.horizon_mae[-1], steps[-1], DISPLAY_NAME.get(r.model, r.model)))

    # Direct labels at the line ends (also the relief for low-contrast hues).
    # Curves that converge -- STGCN and Graph WaveNet do, which is a finding --
    # would otherwise print their labels on top of one another. Stagger the
    # vertical offsets so that a convergence stays readable as a convergence
    # instead of as a typographic collision.
    ends.sort()
    offsets = [0.0] * len(ends)
    min_gap = 11.0  # points; ~1.2 line heights at 9pt
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    pts_per_unit = ax.get_window_extent().height / span if span else 1.0
    for i in range(1, len(ends)):
        gap = (ends[i][0] - ends[i - 1][0]) * pts_per_unit + offsets[i - 1]
        if gap < min_gap:
            offsets[i] = min_gap - gap
    for (y, x, name), dy in zip(ends, offsets):
        ax.annotate(name, (x, y), textcoords='offset points', xytext=(8, dy),
                    fontsize=9, color=INK_PRIMARY, va='center', zorder=4)

    ax.set_xlabel('Forecast horizon (15-min steps)', fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel('Test MAE', fontsize=10, color=INK_SECONDARY)

    # Every line is labelled at its end, so a legend would spell the same four
    # names a second time -- and it lands in the top-left, on top of the data.
    # The gutter on the right exists for those labels, so stop the ticks at
    # the last real horizon rather than tick out into empty space.
    steps = len(records[0].horizon_mae) if records else 12
    ax.set_xlim(0.5, steps + 3)
    ax.set_xticks(range(2, steps + 1, 2))

    fig.tight_layout()
    _save(fig, out)
    return fig


def _save(fig, out):
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(f'{out}.pdf', bbox_inches='tight', facecolor=SURFACE)
    fig.savefig(f'{out}.png', dpi=200, bbox_inches='tight', facecolor=SURFACE)
    print(f'wrote {out}.pdf and {out}.png')


def latex_table(records, grid_intensity=DEFAULT_GRID_INTENSITY, caption=None,
                label='tab:main', epoch_cap=100):
    """The paper's main results table.

    Sorted by training energy so the cost gradient is visible down the column,
    which is the point. FLOPs the counter could not resolve print as '--'
    rather than a fabricated number.

    Models that reach `epoch_cap` never early-stopped, so their training energy
    is a LOWER BOUND and their error is pessimistic. Those rows are daggered
    here rather than left looking like converged runs -- the Threats section
    promises the table flags them, so the table has to actually do it.
    """
    records = sorted(records, key=lambda r: (r.train_joules if np.isfinite(r.train_joules) else 0))
    caption = caption or (
        'Accuracy and cost on LargeST-SD (716 sensors, 2019, 15-min). '
        'Energy is GPU-only, measured via NVML. '
        f'Carbon assumes {grid_intensity:.0f}~gCO$_2$e/kWh. '
        f'$\\dag$~marks a model that reached the {epoch_cap}-epoch cap without '
        'early stopping: its energy is a lower bound, its error pessimistic.')

    def _sig(value, places=2):
        """Keep small numbers legible.

        The lineup spans four orders of magnitude, so a fixed 1-decimal format
        prints NLinear's 0.0035 MFLOPs as '0.0' -- which a reader takes as
        zero cost, and zero cost is precisely the claim the paper must not
        overstate. Widen the format for anything below the rounding floor.
        """
        if not np.isfinite(value):
            return '--'
        if value == 0:
            return '0'
        return f'{value:.{places}f}' if abs(value) >= 10 ** -places else f'{value:.3g}'

    rows = []
    for r in records:
        flops = '--' if not r.flops_reliable or not np.isfinite(r.flops_per_sample) \
            else _sig(r.flops_per_sample / 1e6)
        name = DISPLAY_NAME.get(r.model, r.model) + (
            '$^{\\dag}$' if np.isfinite(r.epochs_run) and r.epochs_run >= epoch_cap
            else '')
        rows.append(' & '.join([
            name,
            f'{r.params:,}',
            flops,
            f'{r.mae:.2f}',
            f'{r.rmse:.2f}',
            f'{r.mape * 100:.2f}',
            _sig(joules_to_kwh(r.train_joules) * 1000),
            _sig(carbon_grams(r.train_joules, grid_intensity)),
            _sig(r.infer_joules_per_1k),
            _sig(r.latency_ms),
        ]) + r' \\')

    return '\n'.join([
        r'\begin{table*}[t]',
        r'\centering',
        r'\caption{' + caption + '}',
        r'\label{' + label + '}',
        r'\begin{tabular}{lrrrrrrrrr}',
        r'\toprule',
        r'Model & Params & MFLOPs & MAE & RMSE & MAPE (\%) & Train (Wh) & gCO$_2$e & Infer (J/1k) & Latency (ms) \\',
        r'\midrule',
        *rows,
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table*}',
    ])
