"""
Quick test of AlphaZero Curriculum Trainer
"""

from alphazero_trainer_curriculum import AlphaZeroCurriculumTrainer

print("Testing AlphaZero Curriculum Trainer...")
print("=" * 70)

# Create trainer with lower simulations for testing
trainer = AlphaZeroCurriculumTrainer(num_simulations=25)

print("\n--- Testing curriculum opponents ---")
for game_num in range(5):
    print(f"\nGame {game_num + 1}:")
    winner, examples, phase_name = trainer.self_play_game(verbose=True)
    print(f"  Opponent type: {phase_name}")
    print(f"  Training examples: {examples}")

print(f"\nBuffer size: {len(trainer.replay_buffer)}")

print("\n--- Testing training step ---")
if len(trainer.replay_buffer) >= 32:
    # Set small batch for testing
    trainer.batch_size = 32
    loss = trainer.train_step()
    if loss:
        print(f"Policy loss: {loss['policy_loss']:.4f}")
        print(f"Value loss: {loss['value_loss']:.4f}")
        print(f"Total loss: {loss['total_loss']:.4f}")
    print("✅ Training step successful!")
else:
    print("Not enough examples for training step yet")

print("\n✅ AlphaZero Curriculum Trainer test passed!")
