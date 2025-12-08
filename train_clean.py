"""Clean training with minimal output - optimized"""

import os
import sys
import io
import time
import torch
import numpy as np
from collections import deque
import argparse

# M2 optimization - better memory management
if torch.backends.mps.is_available():
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Minimal matplotlib config
import matplotlib
matplotlib.use('Agg')

# Windows emoji encoding fix
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
        self._original_stdout, self._original_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = NullWriter(), NullWriter()
        return self

    def __exit__(self, *args):
        sys.stdout, sys.stderr = self._original_stdout, self._original_stderr


# Import env & modules silently
with SuppressOutput():
    from catan_env_pytorch import CatanEnv
    from network_gpu import CatanPolicy
    from agent_gpu import CatanAgent, ExperienceBuffer
    from trainer_gpu import PPOTrainer
    from rule_based_ai import play_rule_based_turn
    from game_system import GameConstants


# --------------------- ARGUMENTS ---------------------
parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=100000)
parser.add_argument('--update-freq', type=int, default=50)
parser.add_argument('--save-freq', type=int, default=10000)
parser.add_argument('--model-name', type=str, default='catan_overnight')
parser.add_argument('--curriculum', action='store_true')
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=50)
args = parser.parse_args()


# --------------------- DEVICE ---------------------
device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"🎮 Device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)} | Batch: {args.batch_size} | Epochs: {args.epochs}")

# --------------------- CURRICULUM ---------------------
CURRICULUM_STAGES = [4, 5, 6, 7, 8, 10]
MASTERY_WINDOW = 100
MASTERY_THRESHOLD = 0.9

if args.curriculum:
    print(f"📚 Adaptive Curriculum ENABLED: {CURRICULUM_STAGES}")
else:
    GameConstants.VICTORY_POINTS_TO_WIN = 10

# --------------------- ENV & AGENT ---------------------
env = CatanEnv(player_id=0)
agent = CatanAgent(device=device)
# Path to your last saved model checkpoint
checkpoint_path = "models/catan_adaptive_final_episode_100000.pt"  # change to your actual file

# Load existing model if checkpoint exists
if os.path.exists(checkpoint_path):
    print(f"🔄 Loading checkpoint: {checkpoint_path}")
    agent.policy.load(checkpoint_path, device=device)

trainer = PPOTrainer(
    model=agent.policy,
    lr=3e-4,            # learning rate
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.1,
    max_grad_norm=0.5,
    device=device
)
buffer = ExperienceBuffer()
# --------------------- TRAINING LOOP ---------------------
episode_rewards = []
episode_vps = deque(maxlen=MASTERY_WINDOW)
current_stage_index = 0
current_vp_target = CURRICULUM_STAGES[current_stage_index]
GameConstants.VICTORY_POINTS_TO_WIN = current_vp_target

timeout_count = 0
natural_end_count = 0
start_time = time.time()

print(f"\nStarting training - Target VP: {current_vp_target}\n")
sys.stdout.flush()

for episode in range(1, args.episodes + 1):
    # Check curriculum mastery
    if args.curriculum and len(episode_vps) == MASTERY_WINDOW:
        if np.mean(episode_vps) >= MASTERY_THRESHOLD * current_vp_target:
            if current_stage_index < len(CURRICULUM_STAGES) - 1:
                current_stage_index += 1
                current_vp_target = CURRICULUM_STAGES[current_stage_index]
                GameConstants.VICTORY_POINTS_TO_WIN = current_vp_target
                print(f"🎉 Mastery achieved! Advancing to VP Target: {current_vp_target}")
                episode_vps.clear()

    # Reset environment
    with SuppressOutput():
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        max_steps = 250 + 50 * max(0, current_vp_target - 5)  # 250, 300, 350

        while not done and step_count < max_steps:
            step_count += 1
            if not info.get('is_my_turn', True):
                play_rule_based_turn(env, env.game_env.game.current_player_index)
                obs = env._get_obs()
                info = env._get_info()
                continue

            # Convert masks once to tensors
            action_mask = torch.tensor(obs['action_mask'], dtype=torch.float32, device=device)
            vertex_mask = torch.tensor(obs['vertex_mask'], dtype=torch.float32, device=device)
            edge_mask = torch.tensor(obs['edge_mask'], dtype=torch.float32, device=device)

            (action, vertex, edge, trade_give, trade_get,
             action_log_prob, vertex_log_prob, edge_log_prob,
             trade_give_log_prob, trade_get_log_prob, value, entropy) = agent.choose_action(
                obs,
                obs['action_mask'],
                obs['vertex_mask'],
                obs['edge_mask']
            )

            next_obs, reward, terminated, truncated, info = env.step(action, vertex, edge, trade_give, trade_get)
            done = terminated or truncated

            buffer.store(
                state=obs['observation'], action=action, vertex=vertex, edge=edge,
                trade_give=trade_give, trade_get=trade_get, reward=reward,
                action_log_prob=action_log_prob, vertex_log_prob=vertex_log_prob,
                edge_log_prob=edge_log_prob, trade_give_log_prob=trade_give_log_prob,
                trade_get_log_prob=trade_get_log_prob, value=value, done=done,
                action_mask=obs['action_mask'], vertex_mask=obs['vertex_mask'], edge_mask=obs['edge_mask']
            )

            obs = next_obs
            episode_reward += reward

        if step_count >= max_steps:
            timeout_count += 1
        else:
            natural_end_count += 1

    # Track metrics
    episode_rewards.append(episode_reward)
    episode_vps.append(info.get('victory_points', 0))

    # Progress print every 100 episodes
    if episode % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        avg_vp = np.mean(episode_vps)
        elapsed = time.time() - start_time
        eps_per_min = episode / (elapsed / 60)
        print(f"[{episode / args.episodes * 100:5.1f}%] Ep {episode:6d}/{args.episodes} | "
              f"VP: {avg_vp:.2f} | Reward: {avg_reward:7.2f} | {eps_per_min:4.0f} eps/min | "
              f"Target VP: {current_vp_target}")
        if episode % 500 == 0:
            print(f"         📊 Timeouts: {timeout_count/episode*100:.1f}% | "
                  f"Natural: {natural_end_count/episode*100:.1f}%")
        sys.stdout.flush()

    # Update policy
    if episode % args.update_freq == 0 and len(buffer) > 0:
        with SuppressOutput():
            metrics = trainer.update_policy(buffer)
        buffer.clear()
        if episode % 500 == 0:
            print(f"         Policy updated | Loss: {metrics['policy_loss']:.4f}")
            sys.stdout.flush()

    # Save checkpoint
    if episode % args.save_freq == 0:
        save_path = f"models/{args.model_name}_episode_{episode}.pt"
        agent.policy.save(save_path)
        print(f"         Checkpoint saved -> {save_path}")
        sys.stdout.flush()

# --------------------- TRAINING SUMMARY ---------------------
elapsed_time = time.time() - start_time
print("\n" + "="*70)
print("TRAINING COMPLETE!")
print(f"Timeout rate: {timeout_count}/{args.episodes} = {timeout_count / args.episodes*100:.1f}%")
print(f"Natural endings: {natural_end_count}/{args.episodes} = {natural_end_count / args.episodes*100:.1f}%")
print(f"Total time: {elapsed_time/60:.1f} min ({elapsed_time/3600:.2f} h) | Avg speed: {args.episodes/(elapsed_time/60):.0f} eps/min")
print("="*70)
