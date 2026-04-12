"""
Training progress visualization for Catantron.

TrainingLogger collects per-game and per-training-step metrics, then
generates a 6-panel matplotlib figure showing learning curves, build
diversity, loss curves, entropy, and curriculum phase progression.

Usage (standalone test):
    python -m training.training_plots
"""

import json
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless servers
    import matplotlib.pyplot as plt
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        pass  # Fallback to default style
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _rolling(data, window=100):
    """Compute rolling average with given window size."""
    if len(data) < window:
        window = max(1, len(data))
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    result = cumsum[window - 1:] / window
    # Pad the beginning with expanding-window average
    prefix = [np.mean(data[:i + 1]) for i in range(min(window - 1, len(data)))]
    return np.concatenate([prefix, result])


class TrainingLogger:
    """Collects training metrics and generates progress plots."""

    def __init__(self):
        # Per-game metrics
        self.game_nums = []
        self.vps = []
        self.wins = []
        self.roads = []
        self.settlements = []
        self.cities = []
        self.dev_cards = []
        self.rewards = []
        self.phases = []

        # Per-training-step metrics
        self.train_game_nums = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.learning_rates = []
        self.entropy_coefs = []

    def log_game(self, game_num, vp, won, roads, settlements, cities, reward, phase_name, dev_cards=0):
        """Record metrics from a completed game."""
        self.game_nums.append(game_num)
        self.vps.append(vp)
        self.wins.append(1 if won else 0)
        self.roads.append(roads)
        self.settlements.append(settlements)
        self.cities.append(cities)
        self.dev_cards.append(dev_cards)
        self.rewards.append(reward)
        self.phases.append(phase_name)

    def log_training(self, game_num, policy_loss, value_loss, entropy, lr, entropy_coef):
        """Record metrics from a training step."""
        self.train_game_nums.append(game_num)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)
        self.learning_rates.append(lr)
        self.entropy_coefs.append(entropy_coef)

    def save_json(self, path):
        """Export all metrics to JSON for later analysis."""
        data = {
            'games': {
                'game_num': self.game_nums,
                'vp': self.vps,
                'win': self.wins,
                'roads': self.roads,
                'settlements': self.settlements,
                'cities': self.cities,
                'dev_cards': self.dev_cards,
                'reward': self.rewards,
                'phase': self.phases,
            },
            'training': {
                'game_num': self.train_game_nums,
                'policy_loss': self.policy_losses,
                'value_loss': self.value_losses,
                'entropy': self.entropies,
                'learning_rate': self.learning_rates,
                'entropy_coef': self.entropy_coefs,
            },
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def plot(self, save_path):
        """Generate a 6-panel training progress figure."""
        if not HAS_MPL:
            print(f"  [TrainingLogger] matplotlib not available, skipping plot")
            return
        if len(self.game_nums) < 2:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Catantron Training Progress', fontsize=16, fontweight='bold')
        games = np.array(self.game_nums)

        # ── Panel 1: VP over time ──
        ax = axes[0, 0]
        vp_roll = _rolling(self.vps, 100)
        ax.plot(games, vp_roll, color='#2196F3', linewidth=1.5)
        ax.set_title('Victory Points (rolling 100)')
        ax.set_xlabel('Games')
        ax.set_ylabel('Avg VP')
        ax.set_ylim(bottom=0)

        # ── Panel 2: Win rate over time ──
        ax = axes[0, 1]
        wr_roll = _rolling(self.wins, 100) * 100
        ax.plot(games, wr_roll, color='#4CAF50', linewidth=1.5)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
        ax.set_title('Win Rate % (rolling 100)')
        ax.set_xlabel('Games')
        ax.set_ylabel('Win Rate %')
        ax.set_ylim(0, 100)
        ax.legend(loc='lower right')

        # ── Panel 3: Build diversity ──
        ax = axes[0, 2]
        road_roll = _rolling(self.roads, 100)
        sett_roll = _rolling(self.settlements, 100)
        city_roll = _rolling(self.cities, 100)
        dev_roll = _rolling(self.dev_cards, 100)
        ax.plot(games, road_roll, color='#795548', linewidth=1.5, label='Roads')
        ax.plot(games, sett_roll, color='#FF9800', linewidth=1.5, label='Settlements')
        ax.plot(games, city_roll, color='#9C27B0', linewidth=1.5, label='Cities')
        ax.plot(games, dev_roll, color='#009688', linewidth=1.5, label='Dev Cards')
        ax.set_title('Build Diversity (rolling 100)')
        ax.set_xlabel('Games')
        ax.set_ylabel('Count per Game')
        ax.legend(loc='upper left')

        # ── Panel 4: Policy loss + Value loss ──
        ax = axes[1, 0]
        if self.train_game_nums:
            tg = np.array(self.train_game_nums)
            ax.plot(tg, self.policy_losses, color='#F44336', linewidth=1.0, alpha=0.8, label='Policy')
            ax2 = ax.twinx()
            ax2.plot(tg, self.value_losses, color='#2196F3', linewidth=1.0, alpha=0.8, label='Value')
            ax2.set_ylabel('Value Loss', color='#2196F3')
            ax.set_ylabel('Policy Loss', color='#F44336')
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax.set_title('Loss Curves')
        ax.set_xlabel('Games')

        # ── Panel 5: Entropy over time ──
        ax = axes[1, 1]
        if self.train_game_nums:
            tg = np.array(self.train_game_nums)
            ax.plot(tg, self.entropies, color='#FF5722', linewidth=1.5)
            ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Healthy min')
            ax.axhline(y=1.8, color='orange', linestyle='--', alpha=0.5, label='Healthy max')
            ax.legend(loc='upper right')
        ax.set_title('Policy Entropy')
        ax.set_xlabel('Games')
        ax.set_ylabel('Entropy')

        # ── Panel 6: Curriculum phase timeline ──
        ax = axes[1, 2]
        if self.phases:
            unique_phases = []
            for p in self.phases:
                if p not in unique_phases:
                    unique_phases.append(p)
            colors = plt.cm.Set3(np.linspace(0, 1, max(len(unique_phases), 1)))
            phase_map = {p: i for i, p in enumerate(unique_phases)}

            # Draw colored bands for each phase
            prev_phase = self.phases[0]
            band_start = games[0]
            for i in range(1, len(self.phases)):
                if self.phases[i] != prev_phase or i == len(self.phases) - 1:
                    pi = phase_map[prev_phase]
                    ax.axvspan(band_start, games[i], alpha=0.4, color=colors[pi])
                    band_start = games[i]
                    prev_phase = self.phases[i]

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=colors[phase_map[p]], alpha=0.4, label=p)
                               for p in unique_phases]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

            # Overlay VP on the phase chart for context
            ax.plot(games, _rolling(self.vps, 100), color='black', linewidth=1.0, alpha=0.7)
            ax.set_ylabel('VP (context)')
        ax.set_title('Curriculum Phases')
        ax.set_xlabel('Games')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [TrainingLogger] Saved plot: {save_path}")


if __name__ == '__main__':
    # Quick self-test with synthetic data
    logger = TrainingLogger()
    for i in range(500):
        phase = 'passive' if i < 100 else ('random' if i < 250 else 'weak')
        vp = min(2 + i * 0.01 + np.random.randn() * 0.5, 10)
        won = vp >= 10 or np.random.random() < (0.1 + i * 0.001)
        logger.log_game(i, vp, won, int(5 + np.random.randn()), 3, int(np.random.random() > 0.5),
                        vp * 10 + np.random.randn() * 5, phase)
        if i % 10 == 0:
            logger.log_training(i, -0.01 + np.random.randn() * 0.005,
                                0.5 - i * 0.0005 + np.random.randn() * 0.05,
                                1.5 - i * 0.002, 5e-4, 0.1 - i * 0.0001)

    logger.plot('/tmp/training_test.png')
    logger.save_json('/tmp/training_test.json')
    print("Self-test complete. Check /tmp/training_test.png")
