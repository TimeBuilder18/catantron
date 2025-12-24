"""Training script that matches YOUR trainer_gpu.py signature"""
import os, sys, io, torch
from collections import deque
import numpy as np
import time

# Fix encoding
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=2000)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--model-name', type=str, default='test')
args = parser.parse_args()

print("="*70)
print("TRAINING - WORKING VERSION")
print("="*70)
print(f"Episodes: {args.episodes} | Epochs: {args.epochs} | Batch: {args.batch_size}")
print()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GameConstants.VICTORY_POINTS_TO_WIN = 4

env = CatanEnv(player_id=0)
agent = CatanAgent(device=device)
print("✅ Fresh agent created\n")

# Use YOUR trainer signature: model, lr, gamma, etc.
trainer = PPOTrainer(
    model=agent.policy,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    entropy_coef=0.05,
    value_coef=0.5,
    max_grad_norm=0.5,
    device=device
)
buffer = ExperienceBuffer()

print("Starting...\n")
start = time.time()
rewards, vps = [], deque(maxlen=100)

for ep in range(args.episodes):
    with SuppressOutput():
        obs, info = env.reset()
        done, step, ep_rew = False, 0, 0

        while not done and step < 250:
            step += 1
            if not info.get('is_my_turn', True):
                play_rule_based_turn(env, env.game_env.game.current_player_index)
                obs, info = env._get_obs(), env._get_info()
                continue

            (act, v, e, tg, tgt, alp, vlp, elp, tglp, tgtp, val) = agent.choose_action(
                obs, obs['action_mask'], obs['vertex_mask'], obs['edge_mask'])

            nobs, rew, term, trunc, info = env.step(act, v, e, tg, tgt)
            done = term or trunc

            buffer.store(obs['observation'], act, v, e, tg, tgt, rew,
                        alp, vlp, elp, tglp, tgtp, val, done,
                        obs['action_mask'], obs['vertex_mask'], obs['edge_mask'])
            obs, ep_rew = nobs, ep_rew + rew

    rewards.append(ep_rew)
    vps.append(info.get('victory_points', 0))

    if (ep + 1) % 100 == 0:
        speed = (ep + 1) / ((time.time() - start) / 60)
        print(f"[{(ep+1)/args.episodes*100:5.1f}%] Ep {ep+1:4d} | "
              f"VP: {np.mean(vps):.2f} | Rew: {np.mean(rewards[-100:]):6.1f} | "
              f"{speed:3.0f} eps/min")

    if (ep + 1) % 50 == 0 and len(buffer) > 0:
        with SuppressOutput():
            trainer.update_policy(buffer)
        buffer.clear()

    if (ep + 1) % 500 == 0:
        os.makedirs("models", exist_ok=True)
        agent.policy.save(f"models/{args.model_name}_ep{ep+1}.pt")
        print(f"         💾 Saved checkpoint")

print(f"\n✅ Done! {(time.time()-start)/60:.1f} min | Final VP: {np.mean(vps):.2f}")
