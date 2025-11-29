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
checkpoint = torch.load(args.model, map_location=device, weights_only=False)

# Handle different checkpoint formats
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    agent.policy.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint format (device: {checkpoint.get('device', 'unknown')})")
else:
    agent.policy.load_state_dict(checkpoint)
    print("Loaded direct state_dict format")

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

action_names = ['roll', 'place_settlement', 'place_road', 'build_settlement', 'build_city', 'build_road',
                'buy_dev', 'end_turn', 'wait', 'trade_with_bank', 'do_nothing']

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

    # Gameplay tracking
    game_log = []

    while not done and step_count < max_steps:
        step_count += 1

        # Get action from agent
        with SuppressOutput():
            (action, vertex, edge, trade_give, trade_get,
             action_log_prob, vertex_log_prob, edge_log_prob,
             trade_give_log_prob, trade_get_log_prob, value) = agent.choose_action(
                obs,
                obs['action_mask'],
                obs['vertex_mask'],
                obs['edge_mask'],
                is_training=False
            )

        # Step environment
        with SuppressOutput():
            next_obs, reward, terminated, truncated, step_info = env.step(action, vertex, edge, trade_give, trade_get)

        done = terminated or truncated
        episode_reward += reward
        obs = next_obs

    # Game finished - get final raw observation
    with SuppressOutput():
        final_raw_obs = env.game_env.get_observation(env.player_id)

    final_vp = final_raw_obs.get('my_victory_points', 0)
    is_timeout = step_count >= max_steps

    # Get final, definitive stats directly from the final observation
    settlements_count = final_raw_obs.get('my_settlements', 0)
    cities_count = final_raw_obs.get('my_cities', 0)
    roads_count = final_raw_obs.get('my_roads', 0)
    dev_cards_count = sum(final_raw_obs.get('my_dev_cards', {}).values())
    
    # Update stats
    game_stats['total_vps'].append(final_vp)
    game_stats['total_rewards'].append(episode_reward)
    game_stats['total_steps'].append(step_count)
    game_stats['cities_built'].append(cities_count)
    game_stats['settlements_built'].append(settlements_count)
    game_stats['roads_built'].append(roads_count)
    game_stats['dev_cards_bought'].append(dev_cards_count)
    
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

print("\n" + "=" * 70)
