"""Training script - NO checkpoint loading"""
import os
import sys
import io
import torch
from collections import deque

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

with SuppressOutput():
    from catan_env_pytorch import CatanEnv
    from network_gpu import CatanPolicy
    from agent_gpu import CatanAgent, ExperienceBuffer
    from trainer_gpu import PPOTrainer
    from rule_based_ai import play_rule_based_turn
    from game_system import GameConstants
    import numpy as np
    import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=2000)
parser.add_argument('--update-freq', type=int, default=50)
parser.add_argument('--save-freq', type=int, default=500)
parser.add_argument('--model-name', type=str, default='test_model')
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()

print("=" * 70)
print("CATAN TRAINING - NO CHECKPOINT LOADING")
print("=" * 70)
print(f"⚠️  Starting FRESH - No checkpoints will be loaded!")
print()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎮 Device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
print(f"   Batch: {args.batch_size} | Epochs: {args.epochs}")
print(f"   Episodes: {args.episodes}")
print()

# Set curriculum
GameConstants.VICTORY_POINTS_TO_WIN = 4
print("🎯 Target VP: 4 (first curriculum stage)\n")

# Create agent (NO loading!)
env = CatanEnv(player_id=0)
agent = CatanAgent(device=device)
print("✅ Created FRESH untrained agent\n")

trainer = PPOTrainer(
    agent.policy,  # positional argument
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.05,
    max_grad_norm=0.5,
    n_epochs=args.epochs,
    batch_size=args.batch_size
)
buffer = ExperienceBuffer()

episode_rewards = []
episode_vps = deque(maxlen=100)

print("Starting training...\n")
sys.stdout.flush()
start_time = time.time()

for episode in range(args.episodes):
    with SuppressOutput():
        obs, info = env.reset()
        done = False
        step_count = 0
        max_steps = 250
        episode_reward = 0

        while not done and step_count < max_steps:
            step_count += 1
            if not info.get('is_my_turn', True):
                current_player = env.game_env.game.current_player_index
                play_rule_based_turn(env, current_player)
                obs = env._get_obs()
                info = env._get_info()
                continue

            (action, vertex, edge, trade_give, trade_get,
             action_log_prob, vertex_log_prob, edge_log_prob,
             trade_give_log_prob, trade_get_log_prob, value) = agent.choose_action(
                obs, obs['action_mask'], obs['vertex_mask'], obs['edge_mask']
            )

            next_obs, reward, terminated, truncated, info = env.step(
                action, vertex, edge, trade_give, trade_get
            )
            done = terminated or truncated

            buffer.store(
                state=obs['observation'], action=action, vertex=vertex, edge=edge,
                trade_give=trade_give, trade_get=trade_get, reward=reward,
                action_log_prob=action_log_prob, vertex_log_prob=vertex_log_prob,
                edge_log_prob=edge_log_prob, trade_give_log_prob=trade_give_log_prob,
                trade_get_log_prob=trade_get_log_prob, value=value, done=done,
                action_mask=obs['action_mask'], vertex_mask=obs['vertex_mask'],
                edge_mask=obs['edge_mask']
            )
            obs = next_obs
            episode_reward += reward

    episode_rewards.append(episode_reward)
    episode_vps.append(info.get('victory_points', 0))

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        avg_vp = np.mean(episode_vps)
        progress = (episode + 1) / args.episodes * 100
        elapsed = time.time() - start_time
        speed = (episode + 1) / (elapsed / 60)

        print(f"[{progress:5.1f}%] Ep {episode + 1:5d}/{args.episodes} | "
              f"VP: {avg_vp:.2f} | Reward: {avg_reward:7.2f} | {speed:4.0f} eps/min")
        sys.stdout.flush()

    if (episode + 1) % args.update_freq == 0 and len(buffer) > 0:
        with SuppressOutput():
            metrics = trainer.update_policy(buffer)
        buffer.clear()

    if (episode + 1) % args.save_freq == 0:
        os.makedirs("models", exist_ok=True)
        save_path = f"models/{args.model_name}_ep{episode + 1}.pt"
        agent.policy.save(save_path)
        print(f"         💾 Saved -> {save_path}")
        sys.stdout.flush()

elapsed_time = time.time() - start_time
print(f"\n✅ Training complete!")
print(f"   Time: {elapsed_time/60:.1f} min")
print(f"   Final VP: {np.mean(episode_vps):.2f}")
print(f"   Speed: {args.episodes / (elapsed_time / 60):.0f} eps/min")
