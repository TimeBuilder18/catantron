"""
Pure PPO training from scratch (no imitation)
"""
from curriculum_trainer_v2_fixed import CurriculumTrainerV2

if __name__ == "__main__":
    print("=" * 70)
    print("PPO TRAINING FROM SCRATCH (NO IMITATION)")
    print("=" * 70)
    print()
    
    # Train from random initialization
    trainer = CurriculumTrainerV2(
        model_path=None,  # Start fresh!
        batch_size=24576,
        learning_rate=3e-4,  # Standard PPO learning rate
        reward_mode='pbrs_fixed'
    )
    
    trainer.train(
        games_per_phase=5000,
        train_frequency=10,
        train_steps=10
    )
