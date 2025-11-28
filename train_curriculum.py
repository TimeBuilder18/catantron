"""
Optimized long training run with curriculum learning
Gradually increases VP target from 5 → 10 as agent improves
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
    from agent_gpu import CatanAgent, ExperienceBuffer
    from trainer_gpu import PPOTrainer
    from rule_based_ai import play_rule_based_turn
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from game_system import GameConstants

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=5000, help='Total episodes to train')
parser.add_argument('--update-freq', type=int, default=50, help='Update policy every N episodes')
parser.add_argument('--save-freq', type=int, default=500, help='Save checkpoint every N episodes')
parser.add_argument('--model-name', type=str, default='catan_curriculum', help='Model name for saving')
parser.add_argument('--resume', type=str, default=None, help='Path to model to resume from')
parser.add_argument('--batch-size', type=int, default=2048, help='Batch size (2048 for RTX 2080 Super)')
parser.add_argument('--epochs', type=int, default=20, help='Training epochs per update')
args = parser.parse_args()

print("=" * 70)
print("CATAN CURRICULUM LEARNING - RTX 2080 SUPER OPTIMIZED")
print("=" * 70)
sys.stdout.flush()

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
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   Batch size: {args.batch_size} (optimized for RTX 2080 Super)")
    print(f"   Training epochs: {args.epochs}")

print(f"\n📚 Curriculum Learning Strategy:")
print(f"   Episodes 0-1000:   VP = 5  (Learn basics)")
print(f"   Episodes 1001-2000: VP = 6  (Build more)")
print(f"   Episodes 2001-3000: VP = 7  (Intermediate)")
print(f"   Episodes 3001-4000: VP = 8  (Advanced)")
print(f"   Episodes 4001+:     VP = 10 (Full game)")

print(f"\n🎯 Training Configuration:")
print(f"   Total episodes: {args.episodes}")
print(f"   Update frequency: {args.update_freq}")
print(f"   Save frequency: {args.save_freq}")
print(f"   Model name: {args.model_name}")
if args.resume:
    print(f"   Resuming from: {args.resume}")
print()
sys.stdout.flush()

env = CatanEnv(player_id=0)
agent = CatanAgent(device=device)

# Resume from checkpoint if specified
start_episode = 0
if args.resume:
    print(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    agent.policy.load_state_dict(checkpoint)
    # Try to infer episode number from filename
    try:
        import re
        match = re.search(r'episode_(\d+)', args.resume)
        if match:
            start_episode = int(match.group(1))
            print(f"Resuming from episode {start_episode}")
    except:
        pass
    print()
    sys.stdout.flush()

# Optimized trainer settings
trainer = PPOTrainer(
    policy=agent.policy,
    learning_rate=3e-4,      # Standard PPO learning rate
    gamma=0.99,              # Discount factor
    gae_lambda=0.95,         # Advantage estimation
    clip_epsilon=0.2,        # PPO clipping
    value_coef=0.5,          # Value loss weight
    entropy_coef=0.05,       # Exploration bonus
    max_grad_norm=0.5,       # Gradient clipping
    n_epochs=args.epochs,
    batch_size=args.batch_size
)
buffer = ExperienceBuffer()

episode_rewards = []
episode_vps = []

print(f"Starting training...\n")
sys.stdout.flush()
start_time = time.time()

timeout_count = 0
natural_end_count = 0

# Curriculum learning: gradually increase VP target
def get_vp_target(episode):
    """Return VP target based on curriculum stage"""
    if episode < 1000:
        return 5
    elif episode < 2000:
        return 6
    elif episode < 3000:
        return 7
    elif episode < 4000:
        return 8
    else:
        return 10

# Training loop
for episode in range(start_episode, args.episodes):
    # Update VP target based on curriculum
    current_vp_target = get_vp_target(episode)
    GameConstants.VICTORY_POINTS_TO_WIN = current_vp_target

    # Collect debug info for periodic printing
    debug_actions = []
    debug_resources = []

    # Suppress game output during episode
    with SuppressOutput():
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        max_steps = 300 if current_vp_target >= 8 else 250  # More steps for higher VP

        # Episode game loop
        while not done and step_count < max_steps:
            step_count += 1

            # Get hierarchical action from agent
            action, vertex, edge, action_log_prob, vertex_log_prob, edge_log_prob, value = agent.choose_action(
                obs,
                obs['action_mask'],
                obs['vertex_mask'],
                obs['edge_mask']
            )

            # Store debug info (only for steps 10-20 during normal play)
            if (episode % args.update_freq == 0 and
                step_count >= 10 and step_count <= 20 and
                obs['game_phase'] == 2):  # NORMAL_PLAY

                valid_actions = ['roll', 'build_settlement', 'build_city',
                               'build_road', 'buy_dev', 'play_knight', 'end']
                action_name = valid_actions[action] if action < len(valid_actions) else 'invalid'

                legal = obs.get('legal_actions', [])
                legal_names = [valid_actions[a] for a in legal if a < len(valid_actions)]

                res = obs['my_resources']
                res_str = f"W{res[0]} B{res[1]} Wh{res[2]} S{res[3]} O{res[4]}"

                debug_actions.append({
                    'step': step_count,
                    'valid': legal_names,
                    'chosen': action_name,
                    'resources': res_str
                })

            # Step environment
            next_obs, reward, terminated, truncated, step_info = env.step(action, vertex, edge)
            done = terminated or truncated

            # Store ALL hierarchical data in buffer
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

            episode_reward += reward
            obs = next_obs

        # Track episode stats
        final_vp = obs.get('my_victory_points', 0)
        episode_rewards.append(episode_reward)
        episode_vps.append(final_vp)

        # Track timeout vs natural ending
        if step_count >= max_steps:
            timeout_count += 1
        else:
            natural_end_count += 1

    # Periodic output
    if (episode + 1) % 10 == 0:
        avg_vp = np.mean(episode_vps[-10:])
        avg_reward = np.mean(episode_rewards[-10:])
        progress = ((episode + 1) / args.episodes) * 100

        elapsed = time.time() - start_time
        eps_per_min = (episode + 1 - start_episode) / (elapsed / 60) if elapsed > 0 else 0

        print(f"[{progress:5.1f}%] Ep {episode+1:5}/{args.episodes} | "
              f"VP: {avg_vp:.1f} | Reward: {avg_reward:7.2f} | "
              f"{eps_per_min:4.0f} eps/min | "
              f"Target VP: {current_vp_target}")
        sys.stdout.flush()

    # Debug output for update episodes
    if (episode + 1) % args.update_freq == 0 and debug_actions:
        print(f"\n  [DEBUG Ep{episode}] Steps 10-20 of normal play:")
        for entry in debug_actions:
            print(f"    Step {entry['step']}: Valid={entry['valid']} | "
                  f"Chose={entry['chosen']} | Res: {entry['resources']}")
        sys.stdout.flush()

    # Update policy
    if (episode + 1) % args.update_freq == 0 and len(buffer) > 0:
        # Calculate timeout percentage for this batch
        total_episodes_so_far = episode + 1
        timeout_pct = (timeout_count / total_episodes_so_far) * 100
        natural_pct = (natural_end_count / total_episodes_so_far) * 100

        print(f"         📊 Timeouts: {timeout_pct:.1f}% | Natural endings: {natural_pct:.1f}%")

        with SuppressOutput():
            policy_loss = trainer.update(buffer)

        print(f"         Policy updated | Loss: {policy_loss:.4f}")
        buffer.clear()
        sys.stdout.flush()

    # Save checkpoint
    if (episode + 1) % args.save_freq == 0:
        os.makedirs('models', exist_ok=True)
        save_path = f"models/{args.model_name}_episode_{episode+1}.pt"
        torch.save(agent.policy.state_dict(), save_path)
        print(f"         Checkpoint saved -> {save_path}\n")
        sys.stdout.flush()

# Final save
os.makedirs('models', exist_ok=True)
final_path = f"models/{args.model_name}_final.pt"
torch.save(agent.policy.state_dict(), final_path)

total_time = time.time() - start_time
avg_speed = args.episodes / (total_time / 60)

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print(f"Final timeout rate: {timeout_count}/{args.episodes} = {(timeout_count/args.episodes)*100:.1f}%")
print(f"Natural endings: {natural_end_count}/{args.episodes} = {(natural_end_count/args.episodes)*100:.1f}%")
print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
print(f"Average speed: {avg_speed:.0f} episodes/min")
print(f"Final model saved -> {final_path}")
print("=" * 70)
