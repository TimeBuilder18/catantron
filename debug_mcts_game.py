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

        # DEBUG: Show game state
        print(f"  DEBUG: can_roll={game.can_roll_dice()}, can_trade={game.can_trade_or_build()}, can_end={game.can_end_turn()}")
        print(f"  DEBUG: game_phase={game.game_phase}, turn_phase={game.turn_phase}")
        print(f"  DEBUG: is_initial={game.is_initial_placement_phase()}, waiting_for_road={game.waiting_for_road}")

        if game.can_roll_dice():
            game.roll_dice()
            print("  Rolled dice")
        elif game.can_trade_or_build():
            # Skip trade/build phase and end turn
            game.end_turn()
            print("  Ended turn (skipped trade/build)")
        elif game.can_end_turn():
            game.end_turn()
            print("  Ended turn")
        else:
            # NOTHING is possible - force advance by ending turn anyway
            print("  WARNING: No valid actions! Forcing end_turn()")
            game.end_turn()

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
