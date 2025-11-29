"""Clean training with minimal output"""
import os
import sys
import io
import os
import torch
from collections import deque

# M2 optimization - better memory management
if torch.backends.mps.is_available():
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    print("✅ M2 memory optimization enabled")
import matplotlib
matplotlib.use('Agg')
# Fix Windows encoding issues with emojis - use reconfigure for Python 3.7+
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    # Fallback for older Python versions
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)


class NullWriter:
    """A file-like object that discards all output - Windows compatible"""
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
        # Use custom NullWriter instead of os.devnull for better Windows compatibility
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
    from agent_gpu import CatanAgent,ExperienceBuffer
    from trainer_gpu import PPOTrainer
    from rule_based_ai import play_rule_based_turn
    from game_system import GameConstants
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import time

# Now run regular training code...
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=100000, help="Total episodes to train for an overnight session")
parser.add_argument('--update-freq', type=int, default=50)
parser.add_argument('--save-freq', type=int, default=10000, help="Save a checkpoint every N episodes")
parser.add_argument('--model-name', type=str, default='catan_overnight')
parser.add_argument('--curriculum', action='store_true', help='Use curriculum learning (VP 4→10)')
parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=20, help='Training epochs per update')
args = parser.parse_args()

print("=" * 70)
print("CATAN TRAINING - ADAPTIVE CURRICULUM")
print("=" * 70)
sys.stdout.flush()

# Adaptive Curriculum Settings
CURRICULUM_STAGES = [4, 5, 6, 7, 8, 10]
MASTERY_WINDOW = 100  # Check average VP over this many episodes
MASTERY_THRESHOLD = 0.9  # Must achieve 90% of target VP to advance

# Auto-detect best device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"🎮 Device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Batch size: {args.batch_size} (RTX 2080 Super optimized)")
    print(f"   Training epochs: {args.epochs}")

if args.curriculum:
    print(f"\n📚 Adaptive Curriculum: ENABLED")
    print(f"   Stages: {CURRICULUM_STAGES}")
    print(f"   Mastery Threshold: {MASTERY_THRESHOLD*100:.0f}% of target VP over {MASTERY_WINDOW} episodes")
else:
    print(f"\n📚 Curriculum Learning: DISABLED")
    GameConstants.VICTORY_POINTS_TO_WIN = 10 # Set to default

print(f"\n🎯 Training:")
print(f"   Episodes: {args.episodes}")
print(f"   Update frequency: {args.update_freq}")
print(f"   Save frequency: {args.save_freq}")
print(f"   Model name: {args.model_name}")
print()
sys.stdout.flush()

env = CatanEnv(player_id=0)
agent = CatanAgent(device=device)

trainer = PPOTrainer(
    policy=agent.policy,
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
episode_vps = deque(maxlen=MASTERY_WINDOW)

current_stage_index = 0
current_vp_target = CURRICULUM_STAGES[current_stage_index]
GameConstants.VICTORY_POINTS_TO_WIN = current_vp_target

print(f"\nStarting training... Initial Target VP: {current_vp_target}\n")
sys.stdout.flush()
start_time = time.time()

timeout_count = 0
natural_end_count = 0

for episode in range(args.episodes):
    if args.curriculum:
        # Check for mastery and advance curriculum
        if len(episode_vps) == MASTERY_WINDOW:
            avg_recent_vp = np.mean(episode_vps)
            if avg_recent_vp >= MASTERY_THRESHOLD * current_vp_target:
                if current_stage_index < len(CURRICULUM_STAGES) - 1:
                    current_stage_index += 1
                    current_vp_target = CURRICULUM_STAGES[current_stage_index]
                    GameConstants.VICTORY_POINTS_TO_WIN = current_vp_target
                    print(f"\n🎉 MASTERY ACHIEVED! Advancing to VP Target: {current_vp_target}\n")
                    episode_vps.clear() # Reset VP window for new stage

    debug_actions = []
    debug_resources = []

    with SuppressOutput():
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        if current_vp_target >= 8:
            max_steps = 350
        elif current_vp_target >= 6:
            max_steps = 300
        else:
            max_steps = 250

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

    episode_rewards.append(episode_reward)
    episode_vps.append(info.get('victory_points', 0))

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        avg_vp = np.mean(episode_vps)
        progress = (episode + 1) / args.episodes * 100
        elapsed = time.time() - start_time
        speed = (episode + 1) / (elapsed / 60)

        progress_str = (
            f"[{progress:5.1f}%] Ep {episode + 1:6d}/{args.episodes} | "
            f"VP: {avg_vp:.2f} | Reward: {avg_reward:7.2f} | {speed:4.0f} eps/min"
        )
        progress_str += f" | Target VP: {current_vp_target}"
        print(progress_str)

        if (episode + 1) % 500 == 0:
            timeout_pct = timeout_count / (episode + 1) * 100
            natural_pct = natural_end_count / (episode + 1) * 100
            print(f"         📊 Timeouts: {timeout_pct:.1f}% | Natural endings: {natural_pct:.1f}%")
        sys.stdout.flush()

    if (episode + 1) % args.update_freq == 0 and len(buffer) > 0:
        with SuppressOutput():
            metrics = trainer.update_policy(buffer)
        buffer.clear()
        if (episode + 1) % 500 == 0:
            print(f"         Policy updated | Loss: {metrics['policy_loss']:.4f}")
            sys.stdout.flush()

    if (episode + 1) % args.save_freq == 0:
        save_path = f"models/{args.model_name}_episode_{episode + 1}.pt"
        agent.policy.save(save_path)
        print(f"         Checkpoint saved -> {save_path}")
        sys.stdout.flush()

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print(f"Final timeout rate: {timeout_count}/{args.episodes} = {timeout_count / args.episodes * 100:.1f}%")
print(f"Natural endings: {natural_end_count}/{args.episodes} = {natural_end_count / args.episodes * 100:.1f}%")
elapsed_time = time.time() - start_time
print(f"Total time: {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.2f} hours)")
print(f"Average speed: {args.episodes / (elapsed_time / 60):.0f} episodes/min")
print("=" * 70)
sys.stdout.flush()