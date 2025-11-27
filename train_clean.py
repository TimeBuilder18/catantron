"""Clean training with minimal output"""
import os
import sys
import io
import os
import torch

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
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import time

# Now run regular training code...
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=30000)
parser.add_argument('--update-freq', type=int, default=50)
parser.add_argument('--save-freq', type=int, default=3000)
parser.add_argument('--model-name', type=str, default='catan_clean')
args = parser.parse_args()

print("=" * 70)
print("CATAN TRAINING - CLEAN OUTPUT MODE")
print("=" * 70)
sys.stdout.flush()

# Auto-detect best device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f" Device: {device}")
print(f"Episodes: {args.episodes}")
print(f"Update frequency: {args.update_freq}")
print(f"Save frequency: {args.save_freq}")
print(f"Model name: {args.model_name}")
sys.stdout.flush()

env = CatanEnv(player_id=0)
agent = CatanAgent(device=device)
trainer = PPOTrainer(
    policy=agent.policy,
    learning_rate=3e-4,
    batch_size=256,      # Reduced from 512 for faster updates
    n_epochs=5,          # Reduced from 10 for speed
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01
)
buffer = ExperienceBuffer()

episode_rewards = []
episode_vps = []

print(f"\nStarting training...\n")
sys.stdout.flush()
start_time = time.time()

for episode in range(args.episodes):
    print(f"Starting episode {episode + 1}...", end='\r')  # ← Add this
    sys.stdout.flush()
    # Suppress game output during episode
    with SuppressOutput():
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        max_steps = 200  # Reduced from 1000 - games should end faster!
        while not done and step_count < max_steps:
            step_count += 1
            if not info.get('is_my_turn', True):
                current_player = env.game_env.game.current_player_index
                play_rule_based_turn(env, current_player)
                obs = env._get_obs()
                info = env._get_info()
                continue

            # Get hierarchical action from agent (7 values now!)
            action, vertex, edge, action_log_prob, vertex_log_prob, edge_log_prob, value = agent.choose_action(
                obs,
                obs['action_mask'],
                obs['vertex_mask'],
                obs['edge_mask']
            )

            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store ALL hierarchical data in buffer (10 values!)
            buffer.store(
                state=obs['observation'],
                action=action,
                vertex=vertex,
                edge=edge,
                reward=reward,
                action_log_prob=action_log_prob,
                vertex_log_prob=vertex_log_prob,
                edge_log_prob=edge_log_prob,
                value=value,
                done=done,
                action_mask=obs['action_mask'],
                vertex_mask=obs['vertex_mask'],
                edge_mask=obs['edge_mask']
            )
            obs = next_obs
            episode_reward += reward

        # Penalty for games that go too long
        if step_count >= max_steps:
            episode_reward -= 50.0  # Penalty for timeout
            done = True

    episode_rewards.append(episode_reward)
    episode_vps.append(info.get('victory_points', 0))
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        avg_vp = np.mean(episode_vps[-10:])
        progress = (episode + 1) / args.episodes * 100
        elapsed = time.time() - start_time
        speed = (episode + 1) / (elapsed / 60)
        print(f"[{progress:5.1f}%] Ep {episode + 1:5d}/{args.episodes} | VP: {avg_vp:.1f} | Reward: {avg_reward:6.2f} | {speed:4.0f} eps/min")
        print(f"[DEBUG] Steps: {step_count}, Reward: {episode_reward:.1f}")

    # Print ONLY every 10 episodes
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        avg_vp = np.mean(episode_vps[-10:])
        progress = (episode + 1) / args.episodes * 100
        elapsed = time.time() - start_time
        speed = (episode + 1) / (elapsed / 60)

        print( f"[{progress:5.1f}%] Ep {episode + 1:5d}/{args.episodes} | VP: {avg_vp:.1f} | Reward: {avg_reward:6.2f} | {speed:4.0f} eps/min")
        sys.stdout.flush()

    # Update policy
    if (episode + 1) % args.update_freq == 0 and len(buffer) > 0:
        with SuppressOutput():
            metrics = trainer.update_policy(buffer)
        buffer.clear()
        print(f"         Policy updated | Loss: {metrics['policy_loss']:.4f}")
        sys.stdout.flush()

    # Save model
    if (episode + 1) % args.save_freq == 0:
        save_path = f"models/{args.model_name}_episode_{episode + 1}.pt"
        agent.policy.save(save_path)
        print(f"         Checkpoint saved -> {save_path}")
        sys.stdout.flush()

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
elapsed_time = time.time() - start_time
print(f"Total time: {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.2f} hours)")
print(f"Average speed: {args.episodes / (elapsed_time / 60):.0f} episodes/min")
print("=" * 70)
sys.stdout.flush()