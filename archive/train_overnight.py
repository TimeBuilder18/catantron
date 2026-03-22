"""
Optimized Overnight PPO Training
- High batch size (8192) for RTX 2080 Super
- PBRS rewards for dense feedback
- 500 games per phase
"""

from curriculum_trainer_v2_fixed import CurriculumTrainerV2

if __name__ == "__main__":
    print("=" * 70)
    print("OVERNIGHT PPO TRAINING - OPTIMIZED")
    print("=" * 70)
    print("Batch size: 8192 (optimized for RTX 2080 Super)")
    print("Reward mode: pbrs_fixed (dense feedback)")
    print("Games per phase: 500")
    print("=" * 70 + "\n")
    
    # Create trainer with high batch size and PBRS rewards
    trainer = CurriculumTrainerV2(
        batch_size=8192,  # HIGH batch size for your GPU!
        reward_mode='pbrs_fixed',  # Dense rewards
        epsilon=0.1,  # 10% exploration
        learning_rate=1e-3
    )
    
    # Train for 2500 games (5 phases × 500)
    trainer.train(games_per_phase=500)
