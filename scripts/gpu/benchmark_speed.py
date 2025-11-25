"""
CPU vs GPU Speed Comparison

Run this to see the speed difference between training on CPU vs GPU.
Perfect for demonstrating why you should use your RTX 2080 Super!
"""

import torch
import time
import sys
sys.path.append('/home/claude')

from catan_env_pytorch import CatanEnv
from agent_gpu import CatanAgent, ExperienceBuffer
from trainer_gpu import PPOTrainer
from rule_based_ai import play_rule_based_turn


def benchmark_device(device_name, episodes=5):
    """Run training on specified device and measure speed"""
    print(f"\n{'='*60}")
    print(f"🧪 Benchmarking: {device_name.upper()}")
    print(f"{'='*60}")
    
    device = torch.device(device_name)
    
    # Create environment and agent
    env = CatanEnv(player_id=0)
    agent = CatanAgent(device=device)
    
    # Create trainer with appropriate batch size
    batch_size = 256 if device.type == 'cuda' else 64
    trainer = PPOTrainer(
        policy=agent.policy,
        batch_size=batch_size,
        n_epochs=10
    )
    
    buffer = ExperienceBuffer()
    
    # Warm-up (first run is slower)
    obs, info = env.reset()
    agent.choose_action(obs, obs['action_mask'])
    
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Wait for GPU to finish warm-up
    
    # Benchmark
    start_time = time.time()
    
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 100:
            # Handle other players
            if not info.get('is_my_turn', True):
                current_player = env.game_env.game.current_player_index
                play_rule_based_turn(env, current_player)
                obs = env._get_obs()
                info = env._get_info()
                continue
            
            # Agent action
            action, log_prob, value = agent.choose_action(obs, obs['action_mask'])
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            buffer.store(
                obs['observation'],
                action,
                reward,
                log_prob,
                value,
                done,
                obs['action_mask']
            )
            
            obs = next_obs
            steps += 1
        
        # Update policy after each episode (for consistent benchmark)
        if len(buffer) > 32:
            trainer.update_policy(buffer)
            buffer.clear()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Wait for GPU to finish
    
    elapsed = time.time() - start_time
    
    print(f"\n📊 Results:")
    print(f"   Episodes: {episodes}")
    print(f"   Total time: {elapsed:.2f}s")
    print(f"   Time per episode: {elapsed/episodes:.2f}s")
    print(f"   Episodes per minute: {(episodes / elapsed) * 60:.1f}")
    
    if device.type == 'cuda':
        mem_allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"   GPU memory used: {mem_allocated:.2f} GB")
    
    return elapsed


def main():
    print("="*60)
    print("⚡ CPU vs GPU Training Speed Benchmark")
    print("="*60)
    print("\nThis will run 5 episodes on each device to compare speed.")
    print("Smaller batch size for CPU, larger for GPU.")
    print("")
    
    # Check what's available
    has_cuda = torch.cuda.is_available()
    
    if has_cuda:
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ No GPU detected - will only benchmark CPU")
    
    input("\nPress Enter to start benchmark...")
    
    # Benchmark CPU
    cpu_time = benchmark_device('cpu', episodes=5)
    
    # Benchmark GPU if available
    if has_cuda:
        gpu_time = benchmark_device('cuda', episodes=5)
        
        print(f"\n{'='*60}")
        print("🏆 COMPARISON")
        print(f"{'='*60}")
        print(f"CPU time:     {cpu_time:.2f}s")
        print(f"GPU time:     {gpu_time:.2f}s")
        print(f"Speedup:      {cpu_time/gpu_time:.1f}x faster on GPU! 🚀")
        print(f"{'='*60}")
        
        # Estimate time for full training
        print(f"\n⏱️  Estimated time for 5000 episodes:")
        print(f"   CPU: {(cpu_time/5) * 5000 / 3600:.1f} hours")
        print(f"   GPU: {(gpu_time/5) * 5000 / 3600:.1f} hours")
        print(f"   Time saved: {((cpu_time - gpu_time)/5) * 5000 / 3600:.1f} hours! 🎉")
    else:
        print(f"\n⚠️  GPU not available. CPU training will work but be ~20-50x slower.")
        print(f"   Estimated time for 5000 episodes: {(cpu_time/5) * 5000 / 3600:.1f} hours")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
