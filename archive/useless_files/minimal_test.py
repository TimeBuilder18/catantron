"""Minimal training test - absolutely no checkpoint loading"""
import sys, os
os.environ['TRAINING_NO_CHECKPOINT'] = '1'  # Flag to prevent any loading

import io
import torch
from collections import deque
import time
import numpy as np

# Fix encoding
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except: pass

class NullWriter:
    def write(self, text): pass
    def flush(self): pass

class SuppressOutput:
    def __enter__(self):
        self._out, self._err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = NullWriter()
        return self
    def __exit__(self, *args):
        sys.stdout, sys.stderr = self._out, self._err

with SuppressOutput():
    from catan_env_pytorch import CatanEnv
    from agent_gpu import CatanAgent, ExperienceBuffer
    from trainer_gpu import PPOTrainer
    from rule_based_ai import play_rule_based_turn
    from game_system import GameConstants

print("="*70)
print("MINIMAL TEST - 2000 EPISODES")
print("="*70)
print("NO checkpoint loading | Target VP: 4 | Epochs: 10\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GameConstants.VICTORY_POINTS_TO_WIN = 4

env = CatanEnv(player_id=0)
agent = CatanAgent(device=device)

trainer = PPOTrainer(
    policy=agent.policy, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
    clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.05,
    max_grad_norm=0.5, n_epochs=10, batch_size=512
)
buffer = ExperienceBuffer()

print("Starting...\n")
start_time = time.time()
episode_rewards, episode_vps = [], deque(maxlen=100)

for episode in range(2000):
    with SuppressOutput():
        obs, info = env.reset()
        done, step_count, episode_reward = False, 0, 0

        while not done and step_count < 250:
            step_count += 1
            if not info.get('is_my_turn', True):
                play_rule_based_turn(env, env.game_env.game.current_player_index)
                obs, info = env._get_obs(), env._get_info()
                continue

            (action, vertex, edge, trade_give, trade_get,
             alp, vlp, elp, tglp, tgtp, value) = agent.choose_action(
                obs, obs['action_mask'], obs['vertex_mask'], obs['edge_mask'])

            next_obs, reward, terminated, truncated, info = env.step(
                action, vertex, edge, trade_give, trade_get)
            done = terminated or truncated

            buffer.store(obs['observation'], action, vertex, edge,
                        trade_give, trade_get, reward, alp, vlp, elp,
                        tglp, tgtp, value, done, obs['action_mask'],
                        obs['vertex_mask'], obs['edge_mask'])
            obs, episode_reward = next_obs, episode_reward + reward

    episode_rewards.append(episode_reward)
    episode_vps.append(info.get('victory_points', 0))

    if (episode + 1) % 100 == 0:
        speed = (episode + 1) / ((time.time() - start_time) / 60)
        print(f"[{(episode+1)/20:5.1f}%] Ep {episode+1:4d} | "
              f"VP: {np.mean(episode_vps):.2f} | "
              f"Rew: {np.mean(episode_rewards[-100:]):6.1f} | "
              f"{speed:3.0f} eps/min")

    if (episode + 1) % 50 == 0 and len(buffer) > 0:
        with SuppressOutput():
            trainer.update_policy(buffer)
        buffer.clear()

    if (episode + 1) % 500 == 0:
        os.makedirs("models", exist_ok=True)
        agent.policy.save(f"models/minimal_test_ep{episode+1}.pt")

elapsed = time.time() - start_time
print(f"\nDone! {elapsed/60:.1f} min | Final VP: {np.mean(episode_vps):.2f}")
