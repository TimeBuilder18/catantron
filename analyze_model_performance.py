"""
Comprehensive Model Performance Analyzer
Analyzes checkpoints to understand learning progression and diagnose issues
"""
import os
import sys
import io
import glob
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Fix encoding
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)


class NullWriter:
    def write(self, text): pass
    def flush(self): pass
    def isatty(self): return False


class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = NullWriter()
        sys.stderr = NullWriter()
        return self

    def __exit__(self, *args):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


# Import with suppression
with SuppressOutput():
    from catan_env_pytorch import CatanEnv
    from network_gpu import CatanPolicy
    from agent_gpu import CatanAgent
    from game_system import GameConstants

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-pattern', type=str, default='models/*episode*.pt',
                    help='Glob pattern for checkpoints')
parser.add_argument('--eval-episodes', type=int, default=20,
                    help='Episodes to evaluate per checkpoint')
parser.add_argument('--vp-target', type=int, default=10,
                    help='Victory points target')
parser.add_argument('--output-dir', type=str, default='analysis_results',
                    help='Directory for output plots and reports')
parser.add_argument('--max-checkpoints', type=int, default=10,
                    help='Maximum number of checkpoints to analyze')
args = parser.parse_args()

print("=" * 80)
print("COMPREHENSIVE MODEL PERFORMANCE ANALYZER")
print("=" * 80)

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Find checkpoints
checkpoints = sorted(glob.glob(args.model_pattern))
if not checkpoints:
    print(f"\n❌ No checkpoints found matching pattern: {args.model_pattern}")
    exit(1)

# Extract episode numbers and sort
checkpoint_data = []
for ckpt_path in checkpoints:
    filename = os.path.basename(ckpt_path)
    match = re.search(r'episode[_-]?(\d+)', filename)
    if match:
        episode_num = int(match.group(1))
        checkpoint_data.append((episode_num, ckpt_path))
    elif 'final' in filename.lower():
        checkpoint_data.append((999999, ckpt_path))  # Sort final last

checkpoint_data.sort()

# Limit number of checkpoints
if len(checkpoint_data) > args.max_checkpoints:
    # Sample evenly across training
    step = len(checkpoint_data) // args.max_checkpoints
    checkpoint_data = [checkpoint_data[i * step] for i in range(args.max_checkpoints)]
    if checkpoint_data[-1][0] != 999999 and any(ep == 999999 for ep, _ in checkpoint_data):
        # Always include final if it exists
        checkpoint_data[-1] = next((ep, path) for ep, path in checkpoint_data if ep == 999999)

print(f"\n📊 Found {len(checkpoints)} checkpoints, analyzing {len(checkpoint_data)}\n")

# Auto-detect device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"🖥️  Device: {device}\n")

# Set VP target
GameConstants.VICTORY_POINTS_TO_WIN = args.vp_target

# Results storage
all_results = []

print("=" * 80)
print("EVALUATING CHECKPOINTS")
print("=" * 80)

for idx, (episode_num, ckpt_path) in enumerate(checkpoint_data):
    print(f"\n[{idx+1}/{len(checkpoint_data)}] Episode {episode_num}: {os.path.basename(ckpt_path)}")
    print("-" * 80)

    # Load model
    env = CatanEnv(player_id=0)
    agent = CatanAgent(device=device)

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            agent.policy.load_state_dict(checkpoint['model_state_dict'])
        else:
            agent.policy.load_state_dict(checkpoint)
        agent.policy.eval()
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        continue

    # Evaluate
    episode_stats = {
        'episode_num': episode_num,
        'checkpoint_path': ckpt_path,
        'vps': [],
        'rewards': [],
        'steps': [],
        'timeouts': 0,
        'natural_endings': 0,
        'cities': [],
        'settlements': [],
        'roads': [],
        'dev_cards': [],
        'max_cards': [],
        'action_counts': defaultdict(int),
        'value_predictions': [],
        'illegal_actions': 0,
    }

    action_names = [
        'roll_dice', 'place_settlement', 'place_road', 'build_settlement',
        'build_city', 'build_road', 'buy_dev_card', 'end_turn',
        'wait', 'trade_with_bank', 'do_nothing'
    ]

    for ep in range(args.eval_episodes):
        with SuppressOutput():
            obs, info = env.reset()

        done = False
        step_count = 0
        max_steps = 500
        episode_reward = 0
        cities_count = 0
        settlements_count = 0
        roads_count = 0
        dev_cards_count = 0
        max_cards_in_game = 0
        illegal_count = 0
        values_in_episode = []

        while not done and step_count < max_steps:
            step_count += 1

            with SuppressOutput():
                (action, vertex, edge, trade_give, trade_get,
                 action_log_prob, vertex_log_prob, edge_log_prob,
                 trade_give_log_prob, trade_get_log_prob, value) = agent.choose_action(
                    obs, obs['action_mask'], obs['vertex_mask'], obs['edge_mask']
                )

                raw_obs = env.game_env.get_observation(env.player_id)

            # Track action
            if action < len(action_names):
                episode_stats['action_counts'][action_names[action]] += 1

            # Track value predictions
            values_in_episode.append(value)

            # Track resources
            total_cards = sum(raw_obs['my_resources'].values())
            max_cards_in_game = max(max_cards_in_game, total_cards)

            # Track building actions
            if action == 1 or action == 3:  # build_settlement
                settlements_count += 1
            elif action == 4:  # build_city
                cities_count += 1
            elif action == 2 or action == 5:  # build_road
                roads_count += 1
            elif action == 6:  # buy_dev_card
                dev_cards_count += 1

            with SuppressOutput():
                next_obs, reward, terminated, truncated, step_info = env.step(
                    action, vertex, edge, trade_give, trade_get
                )

            if step_info.get('illegal_action', False):
                illegal_count += 1

            done = terminated or truncated
            episode_reward += reward
            obs = next_obs

        # Get final state
        with SuppressOutput():
            final_raw_obs = env.game_env.get_observation(env.player_id)

        final_vp = final_raw_obs.get('my_victory_points', 0)
        is_timeout = step_count >= max_steps

        # Store results
        episode_stats['vps'].append(final_vp)
        episode_stats['rewards'].append(episode_reward)
        episode_stats['steps'].append(step_count)
        episode_stats['cities'].append(cities_count)
        episode_stats['settlements'].append(settlements_count)
        episode_stats['roads'].append(roads_count)
        episode_stats['dev_cards'].append(dev_cards_count)
        episode_stats['max_cards'].append(max_cards_in_game)
        episode_stats['value_predictions'].extend(values_in_episode)
        episode_stats['illegal_actions'] += illegal_count

        if is_timeout:
            episode_stats['timeouts'] += 1
        else:
            episode_stats['natural_endings'] += 1

    # Compute summary statistics
    avg_vp = np.mean(episode_stats['vps'])
    std_vp = np.std(episode_stats['vps'])
    avg_reward = np.mean(episode_stats['rewards'])
    avg_steps = np.mean(episode_stats['steps'])
    natural_pct = (episode_stats['natural_endings'] / args.eval_episodes) * 100

    print(f"   VP: {avg_vp:.2f} ± {std_vp:.2f} (target: {args.vp_target})")
    print(f"   Reward: {avg_reward:.1f}")
    print(f"   Steps: {avg_steps:.0f}")
    print(f"   Natural endings: {natural_pct:.0f}%")
    print(f"   Cities: {np.mean(episode_stats['cities']):.1f}")
    print(f"   Settlements: {np.mean(episode_stats['settlements']):.1f}")
    print(f"   Roads: {np.mean(episode_stats['roads']):.1f}")
    print(f"   Illegal actions: {episode_stats['illegal_actions']} total")

    # Top 3 actions
    top_actions = sorted(episode_stats['action_counts'].items(),
                        key=lambda x: x[1], reverse=True)[:3]
    print(f"   Top actions: {', '.join(f'{act}({cnt})' for act, cnt in top_actions)}")

    all_results.append(episode_stats)

# Generate analysis report
print("\n" + "=" * 80)
print("LEARNING PROGRESSION ANALYSIS")
print("=" * 80)

# Extract metrics over training
episodes = [r['episode_num'] for r in all_results]
avg_vps = [np.mean(r['vps']) for r in all_results]
std_vps = [np.std(r['vps']) for r in all_results]
avg_rewards = [np.mean(r['rewards']) for r in all_results]
avg_cities = [np.mean(r['cities']) for r in all_results]
avg_settlements = [np.mean(r['settlements']) for r in all_results]
natural_pcts = [(r['natural_endings'] / args.eval_episodes) * 100 for r in all_results]

# Detect regression
if len(avg_vps) >= 3:
    peak_idx = np.argmax(avg_vps)
    peak_episode = episodes[peak_idx]
    peak_vp = avg_vps[peak_idx]
    final_vp = avg_vps[-1]

    print(f"\n📈 Peak Performance: {peak_vp:.2f} VP at episode {peak_episode}")
    print(f"📊 Final Performance: {final_vp:.2f} VP at episode {episodes[-1]}")

    if final_vp < peak_vp - 0.5:
        regression_pct = ((peak_vp - final_vp) / peak_vp) * 100
        print(f"⚠️  REGRESSION DETECTED: {regression_pct:.1f}% drop from peak")
        print(f"   → Likely cause: Catastrophic forgetting or reward instability")
    else:
        print(f"✅ No major regression detected")

# Detect stagnation
if len(avg_vps) >= 4:
    last_quarter = avg_vps[-len(avg_vps)//4:]
    vp_variance = np.var(last_quarter)
    if vp_variance < 0.1:
        print(f"\n⚠️  STAGNATION DETECTED: Very low variance in recent checkpoints")
        print(f"   → Possible causes: Learning rate too low, stuck in local minimum")

# Detect policy collapse
for r in all_results:
    do_nothing_pct = (r['action_counts']['do_nothing'] /
                     sum(r['action_counts'].values())) * 100 if r['action_counts'] else 0
    if do_nothing_pct > 50:
        print(f"\n⚠️  POLICY COLLAPSE at episode {r['episode_num']}: {do_nothing_pct:.0f}% do_nothing actions")

# Generate plots
print(f"\n📊 Generating plots in {args.output_dir}/")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Model Performance Over Training', fontsize=16)

# Plot 1: VP over time
axes[0, 0].errorbar(episodes, avg_vps, yerr=std_vps, marker='o', capsize=5)
axes[0, 0].axhline(y=args.vp_target, color='r', linestyle='--', label='Target')
axes[0, 0].set_xlabel('Training Episode')
axes[0, 0].set_ylabel('Victory Points')
axes[0, 0].set_title('VP Achievement Over Training')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot 2: Reward over time
axes[0, 1].plot(episodes, avg_rewards, marker='o')
axes[0, 1].set_xlabel('Training Episode')
axes[0, 1].set_ylabel('Average Reward')
axes[0, 1].set_title('Reward Over Training')
axes[0, 1].grid(True)

# Plot 3: Building behavior
axes[0, 2].plot(episodes, avg_settlements, marker='o', label='Settlements')
axes[0, 2].plot(episodes, avg_cities, marker='s', label='Cities')
axes[0, 2].set_xlabel('Training Episode')
axes[0, 2].set_ylabel('Avg Buildings per Game')
axes[0, 2].set_title('Building Strategy Evolution')
axes[0, 2].legend()
axes[0, 2].grid(True)

# Plot 4: Natural endings
axes[1, 0].plot(episodes, natural_pcts, marker='o', color='green')
axes[1, 0].set_xlabel('Training Episode')
axes[1, 0].set_ylabel('Natural Endings (%)')
axes[1, 0].set_title('Game Completion Rate')
axes[1, 0].grid(True)

# Plot 5: Action distribution (latest checkpoint)
latest_actions = all_results[-1]['action_counts']
if latest_actions:
    actions = list(latest_actions.keys())
    counts = list(latest_actions.values())
    axes[1, 1].barh(actions, counts)
    axes[1, 1].set_xlabel('Count')
    axes[1, 1].set_title('Action Distribution (Latest Checkpoint)')

# Plot 6: Value function predictions
value_samples = all_results[-1]['value_predictions'][:1000]  # Sample for visibility
axes[1, 2].hist(value_samples, bins=50, alpha=0.7)
axes[1, 2].set_xlabel('Value Prediction')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Value Function Distribution (Latest)')
axes[1, 2].grid(True)

plt.tight_layout()
plot_path = os.path.join(args.output_dir, 'performance_analysis.png')
plt.savefig(plot_path, dpi=150)
print(f"   Saved: {plot_path}")

# Save detailed report
report_path = os.path.join(args.output_dir, 'analysis_report.txt')
with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DETAILED CHECKPOINT ANALYSIS REPORT\n")
    f.write("=" * 80 + "\n\n")

    for r in all_results:
        f.write(f"\nEpisode {r['episode_num']}:\n")
        f.write(f"  VP: {np.mean(r['vps']):.2f} ± {np.std(r['vps']):.2f}\n")
        f.write(f"  Reward: {np.mean(r['rewards']):.1f}\n")
        f.write(f"  Steps: {np.mean(r['steps']):.0f}\n")
        f.write(f"  Natural endings: {r['natural_endings']}/{args.eval_episodes}\n")
        f.write(f"  Buildings: {np.mean(r['settlements']):.1f} settlements, " +
                f"{np.mean(r['cities']):.1f} cities, {np.mean(r['roads']):.1f} roads\n")
        f.write(f"  Illegal actions: {r['illegal_actions']}\n")
        f.write(f"  Top actions: {dict(sorted(r['action_counts'].items(), key=lambda x: x[1], reverse=True)[:5])}\n")

print(f"   Saved: {report_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\n📁 Results saved to: {args.output_dir}/")
print(f"   - performance_analysis.png (visualization)")
print(f"   - analysis_report.txt (detailed metrics)")
print("\n✅ Analysis complete!")
