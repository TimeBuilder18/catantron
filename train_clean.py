"""Clean training with minimal output"""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', errors='replace')
"""Clean training with minimal output"""
# rest of code...

# Redirect all print statements from imported modules
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, *args):
        sys.stdout.close()
        sys.stderr.close()
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

#print("=" * 70)
#print("CATAN TRAINING - CLEAN OUTPUT MODE")
#print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f"Device: {device}")

env = CatanEnv(player_id=0)
agent = CatanAgent(device=device)
trainer = PPOTrainer(policy=agent.policy, batch_size=512, n_epochs=15)
buffer = ExperienceBuffer()

episode_rewards = []
episode_vps = []

#print(f"\nTraining for {args.episodes} episodes...\n")
start_time = time.time()

for episode in range(args.episodes):
    # Suppress game output during episode
    with SuppressOutput():
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            if not info.get('is_my_turn', True):
                current_player = env.game_env.game.current_player_index
                play_rule_based_turn(env, current_player)
                obs = env._get_obs()
                info = env._get_info()
                continue

            action, log_prob, value = agent.choose_action(obs, obs['action_mask'])
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.store(obs['observation'], action, reward, log_prob, value, done, obs['action_mask'])
            obs = next_obs
            episode_reward += reward

    episode_rewards.append(episode_reward)
    episode_vps.append(info.get('victory_points', 0))

    # Print ONLY every 10 episodes
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        avg_vp = np.mean(episode_vps[-10:])
        progress = (episode + 1) / args.episodes * 100
        elapsed = time.time() - start_time
        speed = (episode + 1) / (elapsed / 60)

        print(
            f"[{progress:5.1f}%] Ep {episode + 1:5d}/{args.episodes} | VP: {avg_vp:.1f} | Reward: {avg_reward:6.2f} | {speed:4.0f} eps/min")

    # Update policy
    if (episode + 1) % args.update_freq == 0 and len(buffer) > 0:
        with SuppressOutput():
            metrics = trainer.update_policy(buffer)
        buffer.clear()
        #print(f"         Policy updated | Loss: {metrics['policy_loss']:.4f}")

    # Save model
    if (episode + 1) % args.save_freq == 0:
        save_path = f"models/{args.model_name}_episode_{episode + 1}.pt"
        agent.policy.save(save_path)
        #print(f"         💾 Checkpoint saved")

#print("\n" + "=" * 70)
#print("TRAINING COMPLETE!")
#print("=" * 70)