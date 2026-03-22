"""
Quick baseline diagnostic - 100 episodes, no training
Just to see what untrained model does
"""
import sys
import io

# Suppress output
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

with SuppressOutput():
    from catan_env_pytorch import CatanEnv
    from agent_gpu import CatanAgent
    from rule_based_ai import play_rule_based_turn
    import torch
    import numpy as np

print("=" * 70)
print("QUICK BASELINE TEST - Untrained Agent Behavior")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

env = CatanEnv(player_id=0)
agent = CatanAgent(device=device)  # Random untrained agent

episodes = 100
action_counts = {}
action_names = [
    'roll_dice', 'place_settlement', 'place_road', 'build_settlement',
    'build_city', 'build_road', 'buy_dev_card', 'end_turn',
    'wait', 'trade_with_bank', 'do_nothing'
]

vps = []
rewards = []
actions_per_turn = []

print(f"Running {episodes} episodes with UNTRAINED agent...\n")

for ep in range(episodes):
    with SuppressOutput():
        obs, info = env.reset()

    done = False
    step_count = 0
    max_steps = 300
    episode_reward = 0
    turn_actions = 0

    while not done and step_count < max_steps:
        step_count += 1

        if not info.get('is_my_turn', True):
            with SuppressOutput():
                play_rule_based_turn(env, env.game_env.game.current_player_index)
                obs = env._get_obs()
                info = env._get_info()
            continue

        with SuppressOutput():
            (action, vertex, edge, trade_give, trade_get, *_) = agent.choose_action(
                obs, obs['action_mask'], obs['vertex_mask'], obs['edge_mask']
            )

        # Track action
        action_name = action_names[action] if action < len(action_names) else 'invalid'
        action_counts[action_name] = action_counts.get(action_name, 0) + 1

        # Track productive actions per turn
        if action_name in ['build_settlement', 'build_city', 'build_road', 'buy_dev_card', 'trade_with_bank']:
            turn_actions += 1
        elif action_name == 'end_turn':
            actions_per_turn.append(turn_actions)
            turn_actions = 0

        with SuppressOutput():
            next_obs, reward, terminated, truncated, step_info = env.step(
                action, vertex, edge, trade_give, trade_get
            )

        done = terminated or truncated
        episode_reward += reward
        obs = next_obs
        info = step_info

    with SuppressOutput():
        final_obs = env.game_env.get_observation(env.player_id)

    vps.append(final_obs.get('my_victory_points', 0))
    rewards.append(episode_reward)

print("=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\n📊 Performance:")
print(f"   Average VP: {np.mean(vps):.2f} (target: 4)")
print(f"   Average Reward: {np.mean(rewards):.1f}")
print(f"   VP Range: {min(vps)} - {max(vps)}")

print(f"\n🎮 Action Distribution:")
total_actions = sum(action_counts.values())
sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
for action, count in sorted_actions[:8]:
    pct = (count / total_actions) * 100
    print(f"   {action:18s}: {count:5d} ({pct:5.1f}%)")

if actions_per_turn:
    print(f"\n🔄 Actions Per Turn:")
    print(f"   Average: {np.mean(actions_per_turn):.2f}")
    print(f"   Max: {max(actions_per_turn)}")
    print(f"   Turns with 0 productive actions: {actions_per_turn.count(0)} ({actions_per_turn.count(0)/len(actions_per_turn)*100:.1f}%)")
    print(f"   Turns with 2+ productive actions: {len([x for x in actions_per_turn if x >= 2])} ({len([x for x in actions_per_turn if x >= 2])/len(actions_per_turn)*100:.1f}%)")

print("\n" + "=" * 70)
print("This shows UNTRAINED random agent behavior.")
print("After training with broken code, agent would learn BAD policies.")
print("After fixes, agent should learn GOOD policies.")
print("=" * 70)
