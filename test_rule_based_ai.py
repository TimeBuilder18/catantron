"""
Test how well the rule-based AI actually performs
"""
import sys
sys.path.append('/mnt/project')

from catan_env_pytorch import CatanEnv
from curriculum_trainer_v2_fixed import play_opponent_turn
from game_system import ResourceType
import random

def test_rule_based_ai(num_games=100):
    """Test rule-based AI win rate vs random opponents"""
    env = CatanEnv(player_id=0)
    wins = 0
    total_vp = 0

    for game_num in range(num_games):
        obs, _ = env.reset()
        done = False
        move_count = 0
        max_moves = 500

        while not done and move_count < max_moves:
            game = env.game_env.game
            current_player_id = game.players.index(game.get_current_player())

            if current_player_id == 0:
                # Rule-based AI plays
                player = game.players[0]
                resources = player.resources
                action_mask = obs['action_mask']
                vertex_mask = obs.get('vertex_mask', [])
                edge_mask = obs.get('edge_mask', [])

                # Use rule-based AI priority
                action_id = None
                vertex_id = 0
                edge_id = 0

                # Priority 1: Build city
                if action_mask[4] == 1:
                    if (resources[ResourceType.WHEAT] >= 2 and
                        resources[ResourceType.ORE] >= 3):
                        valid_vertices = [i for i, m in enumerate(vertex_mask) if m == 1]
                        if valid_vertices:
                            action_id = 4
                            vertex_id = random.choice(valid_vertices)

                # Priority 2: Build settlement
                if action_id is None and action_mask[3] == 1:
                    if (resources[ResourceType.WOOD] >= 1 and
                        resources[ResourceType.BRICK] >= 1 and
                        resources[ResourceType.WHEAT] >= 1 and
                        resources[ResourceType.SHEEP] >= 1):
                        valid_vertices = [i for i, m in enumerate(vertex_mask) if m == 1]
                        if valid_vertices:
                            action_id = 3
                            vertex_id = random.choice(valid_vertices)

                # Priority 3: Build road
                if action_id is None and action_mask[5] == 1:
                    if (resources[ResourceType.WOOD] >= 1 and
                        resources[ResourceType.BRICK] >= 1):
                        valid_edges = [i for i, m in enumerate(edge_mask) if m == 1]
                        if valid_edges:
                            action_id = 5
                            edge_id = random.choice(valid_edges)

                # Priority 4: Buy dev card
                if action_id is None and action_mask[6] == 1:
                    if (resources[ResourceType.WHEAT] >= 1 and
                        resources[ResourceType.SHEEP] >= 1 and
                        resources[ResourceType.ORE] >= 1):
                        action_id = 6

                # Priority 5: End turn
                if action_id is None and action_mask[7] == 1:
                    action_id = 7

                # Priority 6: Roll dice
                if action_id is None and action_mask[0] == 1:
                    action_id = 0

                # Fallback
                if action_id is None:
                    valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                    if valid_actions:
                        action_id = valid_actions[0]
                    else:
                        break

                obs, reward, terminated, truncated, info = env.step(
                    action_id, vertex_id, edge_id,
                    trade_give_idx=0, trade_get_idx=0
                )
                done = terminated or truncated

                if done:
                    vp = info.get('victory_points', 0)
                    total_vp += vp
                    if vp >= 10:
                        wins += 1
            else:
                # Random opponents
                play_opponent_turn(game, current_player_id, random_prob=1.0)

                if game.check_victory_conditions() is not None:
                    done = True
                    # Check if player 0 won
                    vp = game.players[0].get_total_victory_points()
                    total_vp += vp
                    if vp >= 10:
                        wins += 1

            move_count += 1

        if (game_num + 1) % 20 == 0:
            wr = wins / (game_num + 1) * 100
            avg_vp = total_vp / (game_num + 1)
            print(f"Game {game_num+1:3d}/{num_games} | WR: {wr:5.1f}% | Avg VP: {avg_vp:.1f}")

    final_wr = wins / num_games * 100
    final_vp = total_vp / num_games
    print(f"\nFinal Results:")
    print(f"  Win Rate: {final_wr:.1f}%")
    print(f"  Avg VP: {final_vp:.1f}")

if __name__ == "__main__":
    print("Testing Rule-Based AI vs Random Opponents")
    print("="*50)
    test_rule_based_ai(100)
