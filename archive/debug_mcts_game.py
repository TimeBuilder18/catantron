"""
Debug script to see what's happening during MCTS game
"""

import time
from game_state import GameState
from mcts import MCTS
from network_wrapper import NetworkWrapper

print("Creating network wrapper...")
network_wrapper = NetworkWrapper(device='cuda')

print("Creating MCTS with 25 simulations...")
mcts = MCTS(
    policy_network=network_wrapper,
    num_simulations=25,
    c_puct=1.0
)

print("Creating game state...")
state = GameState()

print("\nStarting game simulation...\n")
move_count = 0
max_moves = 100

start_time = time.time()

while not state.is_terminal() and move_count < max_moves:
    current_player = state.get_current_player()

    if current_player == 0:
        move_count += 1
        print(f"Move {move_count}: Player {current_player} thinking with MCTS...")

        search_start = time.time()
        best_action, action_probs = mcts.search(state)
        search_time = time.time() - search_start

        print(f"  MCTS finished in {search_time:.1f}s")
        print(f"  Best action: {best_action}")

        action_id, vertex_id, edge_id = best_action
        reward, done = state.apply_action(action_id, vertex_id, edge_id)

        print(f"  Applied action, reward={reward:.2f}, done={done}")
    else:
        # Random opponent
        print(f"Move {move_count}: Player {current_player} (random opponent)")
        game = state.env.game_env.game
        player = game.players[current_player]

        # Handle initial placement phase
        if game.is_initial_placement_phase():
            import random
            if game.waiting_for_road:
                if game.last_settlement_vertex:
                    edges = game.game_board.edges
                    valid = [e for e in edges
                            if e.structure is None and
                            (e.vertex1 == game.last_settlement_vertex or
                             e.vertex2 == game.last_settlement_vertex)]
                    if valid:
                        game.try_place_initial_road(random.choice(valid), player)
                        print("  Placed initial road")
            else:
                vertices = game.game_board.vertices
                valid = [v for v in vertices if v.structure is None and
                        not any(adj.structure for adj in v.adjacent_vertices)]
                if valid:
                    game.try_place_initial_settlement(random.choice(valid), player)
                    print("  Placed initial settlement")
        # Normal gameplay
        elif game.can_roll_dice():
            game.roll_dice()
            print("  Rolled dice")
        elif game.can_trade_or_build():
            # Skip trade/build phase and end turn
            game.end_turn()
            print("  Ended turn (skipped trade/build)")
        elif game.can_end_turn():
            game.end_turn()
            print("  Ended turn")

    # Check if stuck
    if move_count % 10 == 0:
        vps = [state.get_victory_points(i) for i in range(4)]
        elapsed = time.time() - start_time
        print(f"\n--- After {move_count} moves ({elapsed:.1f}s) ---")
        print(f"Victory Points: {vps}")
        print(f"Terminal: {state.is_terminal()}")
        print()

total_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"Game finished!")
print(f"Total moves: {move_count}")
print(f"Total time: {total_time:.1f}s")
print(f"Winner: {state.get_winner()}")
print(f"Final VPs: {[state.get_victory_points(i) for i in range(4)]}")
print(f"{'='*60}")
