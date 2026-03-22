"""
MCTS Agent for Catan

Uses Monte Carlo Tree Search to choose actions.
Can be used for:
- Playing games (evaluation)
- Generating training data (self-play)
"""

from game_state import GameState
from mcts import MCTS
from network_wrapper import NetworkWrapper


class MCTSAgent:
    """
    Agent that uses MCTS to choose actions
    """

    def __init__(self, model_path=None, num_simulations=50, c_puct=1.0):
        """
        Args:
            model_path: Path to trained model (None = random weights)
            num_simulations: MCTS simulations per move (more = stronger but slower)
            c_puct: Exploration constant
        """
        self.network = NetworkWrapper(model_path=model_path)
        self.mcts = MCTS(
            policy_network=self.network,
            num_simulations=num_simulations,
            c_puct=c_puct
        )
        self.num_simulations = num_simulations

    def choose_action(self, state, temperature=1.0):
        """
        Choose action using MCTS

        Args:
            state: GameState object
            temperature: Controls exploration
                        1.0 = proportional to visit counts (training)
                        0.0 = always pick best (evaluation)

        Returns:
            action: (action_id, vertex_id, edge_id) tuple
            action_probs: dict of action -> probability (for training)
        """
        best_action, action_probs = self.mcts.search(state)

        if temperature == 0:
            # Pick best action
            return best_action, action_probs
        else:
            # Sample from distribution
            import numpy as np
            actions = list(action_probs.keys())
            probs = list(action_probs.values())

            # Apply temperature
            probs = np.array(probs) ** (1.0 / temperature)
            probs = probs / probs.sum()

            # Sample
            idx = np.random.choice(len(actions), p=probs)
            return actions[idx], action_probs

    def play_game(self, verbose=True):
        """
        Play a complete game using MCTS

        Returns:
            winner: Player index who won (0-3)
            history: List of (state, action, action_probs) for training
        """
        from rule_based_ai import play_rule_based_turn

        state = GameState()
        history = []
        move_count = 0
        max_moves = 500  # Prevent infinite games

        if verbose:
            print("Starting game...")

        while not state.is_terminal() and move_count < max_moves:
            current_player = state.get_current_player()

            # Debug: show game phase
            game = state.env.game_env.game
            is_initial = game.is_initial_placement_phase()

            # Check whose turn it is
            if current_player == 0:
                # Our turn - use MCTS
                move_count += 1

                # Debug: show legal actions during initial placement
                if verbose and is_initial and move_count <= 20:
                    obs = state.get_observation()
                    legal_actions = state.get_legal_actions()
                    waiting_road = game.waiting_for_road

                    # Show which edges are marked legal
                    legal_edges = [e for e in range(len(obs['edge_mask'])) if obs['edge_mask'][e] == 1]

                    print(f"  [Init] Move {move_count}: waiting_for_road={waiting_road}")
                    print(f"         Legal actions: {len(legal_actions)}, Legal edges: {legal_edges[:10]}...")

                # Get action from MCTS
                action, action_probs = self.choose_action(state, temperature=1.0)

                # Store for training (state before action, action taken, MCTS policy)
                history.append({
                    'observation': state.get_observation(),
                    'action': action,
                    'action_probs': action_probs,
                    'player': current_player
                })

                # Apply action
                action_id, vertex_id, edge_id = action

                if verbose and is_initial:
                    print(f"  [Init] Chose action={action_id}, vertex={vertex_id}, edge={edge_id}")

                reward, done = state.apply_action(action_id, vertex_id, edge_id)

                if verbose and move_count % 10 == 0 and not is_initial:
                    vps = [state.get_victory_points(i) for i in range(4)]
                    print(f"  Move {move_count}: VPs = {vps}, Action = {action_id}")

            else:
                # Opponent's turn - use rule-based AI
                success = play_rule_based_turn(state.env, current_player)

                if not success:
                    # Fallback: force end turn
                    if state.env.game_env.game.can_end_turn():
                        state.env.game_env.game.end_turn()

            # Check for winner
            if state.is_terminal():
                break

        # Game over
        winner = state.get_winner()

        if verbose:
            if winner is not None:
                print(f"Game over! Winner: Player {winner}")
            else:
                print(f"Game ended (max moves reached)")

            vps = [state.get_victory_points(i) for i in range(4)]
            print(f"Final VPs: {vps}")

        return winner, history


# Quick test
if __name__ == "__main__":
    print("Testing MCTS Agent...")
    print("=" * 50)

    # Create agent with random network (no training yet)
    agent = MCTSAgent(
        model_path=None,  # No trained model
        num_simulations=20,  # Low for fast testing
        c_puct=1.0
    )
    print("✅ Created MCTS Agent")

    # Test choosing an action
    state = GameState()
    action, probs = agent.choose_action(state, temperature=1.0)
    print(f"✅ Chose action: {action}")

    # Play a short game (limit moves for testing)
    print("\n" + "=" * 50)
    print("Playing a test game (this may take a minute)...")
    print("=" * 50)

    # For quick test, reduce simulations
    agent.num_simulations = 10
    agent.mcts.num_simulations = 10

    winner, history = agent.play_game(verbose=True)

    print(f"\n✅ Game complete!")
    print(f"   Total moves: {len(history)}")
    print(f"   Winner: {winner}")

    print("\n✅ MCTS Agent test passed!")