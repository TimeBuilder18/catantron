"""
Evaluate a trained Catan model by watching it play games
Shows detailed game progression and strategy analysis
"""
import os
import sys
import io

# Fix Windows encoding issues with emojis
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)


class NullWriter:
    """A file-like object that discards all output"""
    def write(self, text):
        pass
    def flush(self):
        pass
    def isatty(self):
        return False


class SuppressOutput:
    """Context manager to suppress all stdout/stderr output"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = NullWriter()
        sys.stderr = NullWriter()
        return self

    def __exit__(self, *args):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


# Import everything WHILE suppressing output
with SuppressOutput():
    from catan_env_pytorch import CatanEnv
    from network_gpu import CatanPolicy
    from agent_gpu import CatanAgent
    import torch
    import numpy as np
    from game_system import GameConstants

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
parser.add_argument('--episodes', type=int, default=10, help='Number of games to evaluate')
parser.add_argument('--vp-target', type=int, default=10, help='Victory points needed to win')
parser.add_argument('--verbose', action='store_true', help='Show detailed step-by-step gameplay')
args = parser.parse_args()

print("=" * 70)
print("CATAN MODEL EVALUATION")
print("=" * 70)
print(f"Model: {args.model}")
print(f"Episodes: {args.episodes}")
print(f"VP Target: {args.vp_target}")
print()

# Auto-detect device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Device: {device}")
sys.stdout.flush()

# Set VP target
GameConstants.VICTORY_POINTS_TO_WIN = args.vp_target

# Load model
env = CatanEnv(player_id=0)
agent = CatanAgent(device=device)

print(f"\nLoading model: {args.model}")
checkpoint = torch.load(args.model, map_location=device)
agent.policy.load_state_dict(checkpoint)
agent.policy.eval()  # Set to evaluation mode
print("Model loaded successfully\n")
sys.stdout.flush()

# Statistics tracking
game_stats = {
    'total_vps': [],
    'total_rewards': [],
    'total_steps': [],
    'natural_endings': 0,
    'timeouts': 0,
    'cities_built': [],
    'settlements_built': [],
    'roads_built': [],
    'dev_cards_bought': [],
    'max_cards_held': [],
    'discard_events': [],
}

action_names = ['roll', 'build_settlement', 'build_city', 'build_road',
                'buy_dev', 'play_knight', 'end_turn']

print("Starting evaluation...\n")
print("=" * 70)
sys.stdout.flush()

for episode in range(args.episodes):
    print(f"\n🎮 GAME {episode + 1}/{args.episodes}")
    print("-" * 70)

    with SuppressOutput():
        obs, info = env.reset()

    done = False
    episode_reward = 0
    step_count = 0
    max_steps = 500

    # Track per-game stats
    cities_count = 0
    settlements_count = 0
    roads_count = 0
    dev_cards_count = 0
    max_cards = 0
    discard_count = 0

    # Gameplay tracking
    game_log = []

    while not done and step_count < max_steps:
        step_count += 1

        # Get action from agent
        with SuppressOutput():
            action, vertex, edge, _, _, _, value = agent.choose_action(
                obs,
                obs['action_mask'],
                obs['vertex_mask'],
                obs['edge_mask']
            )

        # Track interesting events
        action_name = action_names[action] if action < len(action_names) else 'invalid'

        # Track resource accumulation
        total_cards = sum(obs['my_resources'].values())
        max_cards = max(max_cards, total_cards)

        # Track discard events
        if obs.get('must_discard', False):
            discard_count += 1

        # Log interesting actions
        if action_name in ['build_city', 'build_settlement', 'build_road', 'buy_dev']:
            my_vp = obs.get('my_victory_points', 0)
            game_log.append({
                'step': step_count,
                'action': action_name,
                'vp': my_vp,
                'cards': total_cards,
                'value': value
            })

            if action_name == 'build_city':
                cities_count += 1
            elif action_name == 'build_settlement':
                settlements_count += 1
            elif action_name == 'build_road':
                roads_count += 1
            elif action_name == 'buy_dev':
                dev_cards_count += 1

        # Step environment
        with SuppressOutput():
            next_obs, reward, terminated, truncated, step_info = env.step(action, vertex, edge)

        done = terminated or truncated
        episode_reward += reward
        obs = next_obs

    # Game finished
    final_vp = obs.get('my_victory_points', 0)
    is_timeout = step_count >= max_steps

    # Update stats
    game_stats['total_vps'].append(final_vp)
    game_stats['total_rewards'].append(episode_reward)
    game_stats['total_steps'].append(step_count)
    game_stats['cities_built'].append(cities_count)
    game_stats['settlements_built'].append(settlements_count)
    game_stats['roads_built'].append(roads_count)
    game_stats['dev_cards_bought'].append(dev_cards_count)
    game_stats['max_cards_held'].append(max_cards)
    game_stats['discard_events'].append(discard_count)

    if is_timeout:
        game_stats['timeouts'] += 1
    else:
        game_stats['natural_endings'] += 1

    # Print game summary
    print(f"Final VP: {final_vp}/{args.vp_target}")
    print(f"Total Reward: {episode_reward:.1f}")
    print(f"Steps: {step_count}")
    print(f"Ending: {'⏱️ Timeout' if is_timeout else '✅ Natural'}")
    print(f"\nBuildings: {settlements_count} settlements, {cities_count} cities, {roads_count} roads")
    print(f"Dev Cards: {dev_cards_count} bought")
    print(f"Max Cards Held: {max_cards} (Discard Events: {discard_count})")

    if game_log and args.verbose:
        print(f"\nKey Actions:")
        for entry in game_log[:15]:  # Show first 15 key actions
            print(f"  Step {entry['step']:3d}: {entry['action']:16s} | VP: {entry['vp']} | Cards: {entry['cards']} | Value: {entry['value']:.2f}")

    sys.stdout.flush()

# Final summary
print("\n" + "=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

avg_vp = np.mean(game_stats['total_vps'])
avg_reward = np.mean(game_stats['total_rewards'])
avg_steps = np.mean(game_stats['total_steps'])
natural_pct = (game_stats['natural_endings'] / args.episodes) * 100
timeout_pct = (game_stats['timeouts'] / args.episodes) * 100

print(f"\n📊 Overall Performance:")
print(f"   Average VP: {avg_vp:.2f} / {args.vp_target}")
print(f"   Average Reward: {avg_reward:.1f}")
print(f"   Average Steps: {avg_steps:.0f}")
print(f"   Natural Endings: {game_stats['natural_endings']}/{args.episodes} ({natural_pct:.1f}%)")
print(f"   Timeouts: {game_stats['timeouts']}/{args.episodes} ({timeout_pct:.1f}%)")

print(f"\n🏗️ Building Statistics:")
print(f"   Settlements: {np.mean(game_stats['settlements_built']):.1f} per game (total: {sum(game_stats['settlements_built'])})")
print(f"   Cities: {np.mean(game_stats['cities_built']):.1f} per game (total: {sum(game_stats['cities_built'])})")
print(f"   Roads: {np.mean(game_stats['roads_built']):.1f} per game (total: {sum(game_stats['roads_built'])})")
print(f"   Dev Cards: {np.mean(game_stats['dev_cards_bought']):.1f} per game (total: {sum(game_stats['dev_cards_bought'])})")

print(f"\n🃏 Resource Management:")
print(f"   Max Cards Held: {np.mean(game_stats['max_cards_held']):.1f} average")
print(f"   Discard Events: {np.mean(game_stats['discard_events']):.1f} per game (total: {sum(game_stats['discard_events'])})")
print(f"   Games with Discards: {sum(1 for d in game_stats['discard_events'] if d > 0)}/{args.episodes}")

print("\n" + "=" * 70)
