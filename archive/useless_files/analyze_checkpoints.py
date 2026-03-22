"""
Analyze and compare model checkpoints from curriculum learning
Shows how the agent improved across different training stages
"""
import os
import glob

print("=" * 70)
print("CURRICULUM LEARNING CHECKPOINT ANALYSIS")
print("=" * 70)

# Find all curriculum checkpoints
checkpoints = sorted(glob.glob("models/catan_curriculum_episode_*.pt"))
final_model = "models/catan_curriculum_final.pt"

if os.path.exists(final_model):
    checkpoints.append(final_model)

if not checkpoints:
    print("\n❌ No checkpoints found in models/ directory")
    print("   Expected files like: models/catan_curriculum_episode_*.pt")
    exit(1)

print(f"\n📊 Found {len(checkpoints)} checkpoints:\n")

for checkpoint in checkpoints:
    filename = os.path.basename(checkpoint)
    size_mb = os.path.getsize(checkpoint) / (1024 * 1024)
    modified_time = os.path.getmtime(checkpoint)

    # Extract episode number
    if "final" in filename:
        episode = "Final"
        vp_stage = "10 VP"
    else:
        import re
        match = re.search(r'episode_(\d+)', filename)
        if match:
            ep_num = int(match.group(1))
            episode = f"{ep_num:5d}"

            # Determine curriculum stage
            if ep_num <= 1000:
                vp_stage = "5 VP"
            elif ep_num <= 2000:
                vp_stage = "6 VP"
            elif ep_num <= 3000:
                vp_stage = "7 VP"
            elif ep_num <= 4000:
                vp_stage = "8 VP"
            else:
                vp_stage = "10 VP"
        else:
            episode = "Unknown"
            vp_stage = "?"

    print(f"   Episode {episode} | Stage: {vp_stage} | Size: {size_mb:.1f} MB | {filename}")

print("\n" + "=" * 70)
print("TRAINING PROGRESSION")
print("=" * 70)

print("""
Based on curriculum learning schedule:

Episodes    0-1000:  VP Target = 5  (Learn basics)
  → Focus: Resource collection, basic building, turn management
  → Expected: Learn to build settlements and roads, basic resource trading

Episodes 1001-2000:  VP Target = 6  (Build more)
  → Focus: Expanding settlements, strategic positioning
  → Expected: More consistent building patterns, better resource management

Episodes 2001-3000:  VP Target = 7  (Intermediate)
  → Focus: City building, development cards
  → Expected: Agent starts building cities, using dev cards strategically

Episodes 3001-4000:  VP Target = 8  (Advanced)
  → Focus: Long-term planning, complex strategies
  → Expected: Multi-turn planning, efficient resource usage

Episodes 4001-5000:  VP Target = 10 (Full game)
  → Focus: Winning strategies, optimal play
  → Expected: Competitive gameplay, strategic card management
""")

print("=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

print("""
To evaluate the checkpoints:

1. Test early checkpoint (episode 1000):
   python evaluate_model.py --model models/catan_curriculum_episode_1000.pt --episodes 10 --vp-target 5

2. Test mid-training checkpoint (episode 3000):
   python evaluate_model.py --model models/catan_curriculum_episode_3000.pt --episodes 10 --vp-target 7

3. Test final model:
   python evaluate_model.py --model models/catan_curriculum_final.pt --episodes 10 --vp-target 10

4. Compare performance across checkpoints to see learning progression

5. Run longer evaluation for more accurate statistics:
   python evaluate_model.py --model models/catan_curriculum_final.pt --episodes 50 --vp-target 10
""")

print("=" * 70)
