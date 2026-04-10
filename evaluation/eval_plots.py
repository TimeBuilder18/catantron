"""
Evaluation benchmark visualization for Catantron.

Generates publication-quality plots from benchmark results:
  - plot_benchmark(): 6-panel summary across all opponent levels
  - plot_single_opponent(): 3-panel detail for one opponent

Usage (standalone test):
    python -m evaluation.eval_plots
"""

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        pass
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_benchmark(all_results, save_path):
    """
    Generate a 6-panel benchmark summary from evaluate() results.

    Args:
        all_results: dict of {opponent_name: result_dict} from benchmark()
        save_path: path to save PNG
    """
    if not HAS_MPL:
        print(f"  [eval_plots] matplotlib not available, skipping plot")
        return
    if not all_results:
        return

    opponents = list(all_results.keys())
    n = len(opponents)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Catantron Benchmark Results', fontsize=16, fontweight='bold')
    x = np.arange(n)
    bar_width = 0.6

    # ── Panel 1: Win rate by opponent ──
    ax = axes[0, 0]
    win_rates = [all_results[o]['win_rate'] for o in opponents]
    bars = ax.bar(x, win_rates, bar_width, color='#4CAF50', edgecolor='white')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Win Rate by Opponent')
    ax.set_ylabel('Win Rate %')
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(opponents, rotation=30, ha='right', fontsize=9)
    for bar, val in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=9)

    # ── Panel 2: AQS by opponent ──
    ax = axes[0, 1]
    aqs_scores = [all_results[o]['aqs'] for o in opponents]
    bars = ax.bar(x, aqs_scores, bar_width, color='#2196F3', edgecolor='white')
    ax.set_title('Agent Quality Score')
    ax.set_ylabel('AQS (0-1000)')
    ax.set_ylim(0, max(1000, max(aqs_scores) * 1.1) if aqs_scores else 1000)
    ax.set_xticks(x)
    ax.set_xticklabels(opponents, rotation=30, ha='right', fontsize=9)
    for bar, val in zip(bars, aqs_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    # ── Panel 3: Average VP (model vs opponent) ──
    ax = axes[0, 2]
    model_vps = [all_results[o]['model_avg_vp'] for o in opponents]
    opp_vps = [all_results[o]['opp_avg_vp'] for o in opponents]
    w = 0.35
    ax.bar(x - w / 2, model_vps, w, color='#4CAF50', label='Model', edgecolor='white')
    ax.bar(x + w / 2, opp_vps, w, color='#F44336', label='Opponent', edgecolor='white')
    ax.set_title('Average VP: Model vs Opponent')
    ax.set_ylabel('Victory Points')
    ax.set_xticks(x)
    ax.set_xticklabels(opponents, rotation=30, ha='right', fontsize=9)
    ax.legend()

    # ── Panel 4: Build diversity (stacked bar) ──
    ax = axes[1, 0]
    roads = [all_results[o]['model_avg_roads'] for o in opponents]
    setts = [all_results[o]['model_avg_sett'] for o in opponents]
    cities = [all_results[o]['model_avg_cities'] for o in opponents]
    ax.bar(x, roads, bar_width, color='#795548', label='Roads', edgecolor='white')
    ax.bar(x, setts, bar_width, bottom=roads, color='#FF9800', label='Settlements', edgecolor='white')
    bottoms = [r + s for r, s in zip(roads, setts)]
    ax.bar(x, cities, bar_width, bottom=bottoms, color='#9C27B0', label='Cities', edgecolor='white')
    ax.set_title('Build Diversity by Opponent')
    ax.set_ylabel('Avg Count per Game')
    ax.set_xticks(x)
    ax.set_xticklabels(opponents, rotation=30, ha='right', fontsize=9)
    ax.legend(loc='upper right')

    # ── Panel 5: Action distribution heatmap ──
    ax = axes[1, 1]
    # Collect all action types across opponents
    all_actions = set()
    for o in opponents:
        all_actions.update(all_results[o].get('action_pcts', {}).keys())
    # Filter to interesting actions (skip roll_dice, wait, do_nothing)
    skip = {'roll_dice', 'wait', 'do_nothing'}
    action_names = sorted(a for a in all_actions if a not in skip)
    if action_names:
        matrix = np.zeros((n, len(action_names)))
        for i, o in enumerate(opponents):
            pcts = all_results[o].get('action_pcts', {})
            for j, a in enumerate(action_names):
                matrix[i, j] = pcts.get(a, 0)
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')
        ax.set_yticks(range(n))
        ax.set_yticklabels(opponents, fontsize=9)
        ax.set_xticks(range(len(action_names)))
        ax.set_xticklabels(action_names, rotation=45, ha='right', fontsize=8)
        fig.colorbar(im, ax=ax, label='% of actions', shrink=0.8)
    ax.set_title('Action Distribution')

    # ── Panel 6: Game length ──
    ax = axes[1, 2]
    avg_moves = [all_results[o]['avg_moves'] for o in opponents]
    bars = ax.bar(x, avg_moves, bar_width, color='#607D8B', edgecolor='white')
    ax.set_title('Average Game Length')
    ax.set_ylabel('Moves')
    ax.set_xticks(x)
    ax.set_xticklabels(opponents, rotation=30, ha='right', fontsize=9)
    for bar, val in zip(bars, avg_moves):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [eval_plots] Saved benchmark plot: {save_path}")


def plot_single_opponent(result, save_path):
    """
    Generate a 3-panel detail plot for a single opponent evaluation.

    Args:
        result: result dict from evaluate()
        save_path: path to save PNG
    """
    if not HAS_MPL:
        print(f"  [eval_plots] matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    opp_name = result.get('opponent', 'unknown')
    fig.suptitle(f'Catantron vs {opp_name.upper()}', fontsize=14, fontweight='bold')

    # ── Panel 1: AQS breakdown (horizontal bar) ──
    ax = axes[0]
    breakdown = result.get('aqs_breakdown', {})
    components = breakdown.get('components', {})
    if components:
        names = list(components.keys())
        scores = [components[n]['score'] for n in names]
        weights = [components[n]['weight'] for n in names]
        weighted = [components[n]['weighted'] for n in names]
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, scores, color=colors, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        for bar, w, ws in zip(bars, weights, weighted):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f'x{w:.1f}={ws:.0f}', va='center', fontsize=8)
    ax.set_title(f'AQS Breakdown ({breakdown.get("aqs", 0):.0f}/1000)')
    ax.set_xlabel('Raw Score')

    # ── Panel 2: Action distribution (pie chart) ──
    ax = axes[1]
    action_pcts = result.get('action_pcts', {})
    if action_pcts:
        # Filter out tiny slices for readability
        filtered = {k: v for k, v in action_pcts.items() if v >= 1.0}
        other = sum(v for v in action_pcts.values() if v < 1.0)
        if other > 0:
            filtered['other'] = other
        labels = list(filtered.keys())
        sizes = list(filtered.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
               textprops={'fontsize': 8}, startangle=90)
    ax.set_title('Action Distribution')

    # ── Panel 3: Structure counts (bar) ──
    ax = axes[2]
    categories = ['Roads', 'Settlements', 'Cities', 'Dev Cards', 'Knights']
    model_vals = [
        result.get('model_avg_roads', 0),
        result.get('model_avg_sett', 0),
        result.get('model_avg_cities', 0),
        result.get('model_avg_dev', 0),
        result.get('model_avg_knights', 0),
    ]
    opp_vals = [
        result.get('opp_avg_roads', 0),
        result.get('opp_avg_sett', 0),
        result.get('opp_avg_cities', 0),
        0,  # Opponent dev cards not tracked
        result.get('opp_avg_knights', 0),
    ]
    x_pos = np.arange(len(categories))
    w = 0.35
    ax.bar(x_pos - w / 2, model_vals, w, color='#4CAF50', label='Model', edgecolor='white')
    ax.bar(x_pos + w / 2, opp_vals, w, color='#F44336', label='Opponent', edgecolor='white')
    ax.set_title('Structure & Card Counts')
    ax.set_ylabel('Avg per Game')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=30, ha='right', fontsize=9)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [eval_plots] Saved detail plot: {save_path}")


if __name__ == '__main__':
    # Quick self-test with synthetic data
    fake_results = {}
    for opp in ['passive', 'random', 'weak', 'medium', 'strong']:
        difficulty = ['passive', 'random', 'weak', 'medium', 'strong'].index(opp)
        wr = max(10, 90 - difficulty * 18)
        fake_results[opp] = {
            'opponent': opp, 'games': 50,
            'win_rate': wr, 'opp_win_rate': 100 - wr - 5, 'timeout_rate': 5,
            'model_avg_vp': 6 + (4 - difficulty) * 0.8, 'model_std_vp': 2.0,
            'opp_avg_vp': 3 + difficulty * 1.2,
            'avg_moves': 80 + difficulty * 20, 'max_moves': 200,
            'model_avg_sett': 3.5, 'model_avg_cities': 1.2, 'model_avg_roads': 7.0,
            'opp_avg_sett': 2.8, 'opp_avg_cities': 0.8, 'opp_avg_roads': 6.0,
            'model_avg_knights': 1.5, 'model_largest_pct': 40, 'model_longest_pct': 35,
            'opp_avg_knights': 1.0, 'opp_largest_pct': 20, 'opp_longest_pct': 25,
            'model_avg_dev': 2.5,
            'action_pcts': {
                'end_turn': 25, 'build_road': 15, 'build_settlement': 8,
                'build_city': 5, 'buy_dev_card': 7, 'trade_with_bank': 10,
                'play_knight': 3, 'place_settlement': 5, 'place_road': 5,
            },
            'aqs': 450 + (4 - difficulty) * 80,
            'aqs_tier': 'Good',
            'aqs_breakdown': {
                'aqs': 450 + (4 - difficulty) * 80,
                'tier': 'Good',
                'components': {
                    'win_rate': {'score': wr, 'weight': 3.0, 'weighted': wr * 3},
                    'vp_efficiency': {'score': 60, 'weight': 2.0, 'weighted': 120},
                    'build_quality': {'score': 55, 'weight': 1.5, 'weighted': 82.5},
                    'resource_mgmt': {'score': 50, 'weight': 1.0, 'weighted': 50},
                },
            },
            'games_per_min': 30,
        }

    plot_benchmark(fake_results, '/tmp/benchmark_test.png')
    plot_single_opponent(fake_results['medium'], '/tmp/detail_test.png')
    print("Self-test complete. Check /tmp/benchmark_test.png and /tmp/detail_test.png")
