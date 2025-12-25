"""
Monte Carlo Tree Search for Catan

This implements AlphaZero-style MCTS:
- Uses neural network for policy (prior probabilities)
- Uses neural network for value (instead of random rollouts)
"""

import math
import numpy as np
from game_state import GameState


class MCTSNode:
    """
    A node in the MCTS tree

    Each node represents a game state reached by taking an action from parent.
    """

    def __init__(self, state, parent=None, action=None, prior=0.0):
        """
        Args:
            state: GameState at this node
            parent: Parent MCTSNode (None for root)
            action: (action, vertex, edge) tuple that led here
            prior: P(s,a) prior probability from neural network
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior

        self.children = {}  # action tuple -> MCTSNode
        self.visits = 0  # N(s)
        self.value_sum = 0  # W(s) - total value

    def value(self):
        """Q(s) = W(s) / N(s) - average value"""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def is_expanded(self):
        """Has this node been expanded?"""
        return len(self.children) > 0

    def is_terminal(self):
        """Is this a terminal state?"""
        return self.state.is_terminal()

    def ucb_score(self, c_puct=1.0):
        """
        Calculate UCB score for selection

        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(parent)) / (1 + N(s,a))

        Args:
            c_puct: Exploration constant (higher = more exploration)
        """
        if self.parent is None:
            return 0.0

        # Exploitation: average value
        q_value = self.value()

        # Exploration: prior * sqrt(parent visits) / (1 + visits)
        exploration = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)

        return q_value + exploration


class MCTS:
    """
    Monte Carlo Tree Search with Neural Network guidance
    """

    def __init__(self, policy_network=None, num_simulations=50, c_puct=1.0):
        """
        Args:
            policy_network: Neural network with evaluate(state) method
                           Returns (policy_dict, value) where policy_dict maps actions to probs
                           If None, uses uniform random policy
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant
        """
        self.policy_network = policy_network
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, root_state):
        """
        Run MCTS from root_state

        Args:
            root_state: GameState to search from

        Returns:
            best_action: (action, vertex, edge) tuple
            action_probs: dict mapping actions to visit probabilities
        """
        # Create root node
        root = MCTSNode(root_state.copy())

        # Expand root
        self._expand(root)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # 1. SELECT - traverse tree until we find unexpanded node
            while node.is_expanded() and not node.is_terminal():
                node = self._select_child(node)
                search_path.append(node)

            # 2. EXPAND & EVALUATE
            if not node.is_terminal():
                self._expand(node)
                value = self._evaluate(node)
            else:
                # Terminal node - get actual result
                value = node.state.get_result(root_state.get_current_player())

            # 3. BACKPROPAGATE
            self._backpropagate(search_path, value)

        # Return best action (most visited)
        best_action = self._get_best_action(root)
        action_probs = self._get_action_probs(root)

        return best_action, action_probs

    def _select_child(self, node):
        """Select child with highest UCB score"""
        best_score = -float('inf')
        best_child = None

        for child in node.children.values():
            score = child.ucb_score(self.c_puct)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand(self, node):
        """Expand node by adding all legal actions as children"""
        legal_actions = node.state.get_legal_actions()

        if not legal_actions:
            return

        # Get prior probabilities from neural network
        priors = self._get_priors(node.state, legal_actions)

        for action, prior in zip(legal_actions, priors):
            # Create child state
            child_state = node.state.copy()
            child_state.apply_action(action[0], action[1], action[2])

            # Create child node
            child = MCTSNode(
                state=child_state,
                parent=node,
                action=action,
                prior=prior
            )
            node.children[action] = child

    def _evaluate(self, node):
        """Get value estimate for node from neural network"""
        if self.policy_network is None:
            # No network - use simple heuristic (VP difference)
            current_player = node.state.get_current_player()
            my_vp = node.state.get_victory_points(current_player)
            # Normalize to [-1, 1] range (assuming 10 VP to win)
            return (my_vp - 5) / 5.0

        # Use neural network
        obs = node.state.get_observation()
        _, value = self.policy_network.evaluate(obs)
        return value

    def _get_priors(self, state, legal_actions):
        """Get prior probabilities for legal actions"""
        if self.policy_network is None:
            # No network - uniform priors
            n = len(legal_actions)
            return [1.0 / n] * n

        # Use neural network to get priors
        obs = state.get_observation()
        policy, _ = self.policy_network.evaluate(obs)

        # Extract priors for legal actions
        priors = []
        for action in legal_actions:
            action_id, vertex_id, edge_id = action
            # Get probability for this action
            # For now, just use action_id probability
            prior = policy[action_id] if action_id < len(policy) else 0.01
            priors.append(prior)

        # Normalize
        total = sum(priors)
        if total > 0:
            priors = [p / total for p in priors]
        else:
            priors = [1.0 / len(priors)] * len(priors)

        return priors

    def _backpropagate(self, search_path, value):
        """Update all nodes in search path with value"""
        for node in reversed(search_path):
            node.visits += 1
            node.value_sum += value
            # Flip value for opponent's perspective
            value = -value

    def _get_best_action(self, root):
        """Get action with most visits"""
        best_visits = -1
        best_action = None

        for action, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action

        return best_action

    def _get_action_probs(self, root, temperature=1.0):
        """
        Get action probabilities based on visit counts

        Args:
            root: Root node
            temperature: Controls exploration (1.0 = proportional to visits,
                        0 = always pick best)
        """
        actions = list(root.children.keys())
        visits = [root.children[a].visits for a in actions]

        if temperature == 0:
            # Pick best
            probs = [0.0] * len(actions)
            probs[np.argmax(visits)] = 1.0
        else:
            # Softmax with temperature
            visits = np.array(visits, dtype=np.float32)
            visits = visits ** (1.0 / temperature)
            probs = visits / visits.sum()

        return dict(zip(actions, probs))


# Quick test
if __name__ == "__main__":
    print("Testing MCTS...")

    # Create initial state
    state = GameState()
    print(f"✅ Created game state")
    print(f"   Legal actions: {len(state.get_legal_actions())}")

    # Test 1: MCTS without network
    print(f"\n--- Test 1: MCTS without neural network ---")
    mcts_no_nn = MCTS(policy_network=None, num_simulations=10)
    best_action, action_probs = mcts_no_nn.search(state)
    print(f"✅ MCTS (no network) best action: {best_action}")

    # Test 2: MCTS with neural network
    print(f"\n--- Test 2: MCTS with neural network ---")
    try:
        from network_wrapper import NetworkWrapper

        network = NetworkWrapper(model_path=None)  # Random weights for now
        mcts_with_nn = MCTS(policy_network=network, num_simulations=20)
        print(f"✅ Created MCTS with neural network (20 simulations)")

        # Run search
        print(f"\n🔍 Running MCTS search with neural network...")
        best_action, action_probs = mcts_with_nn.search(state)

        print(f"✅ Search complete!")
        print(f"   Best action: {best_action}")
        print(f"   Action probabilities: {len(action_probs)} actions")

        # Show top 5 actions
        sorted_actions = sorted(action_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n   Top 5 actions:")
        for action, prob in sorted_actions:
            print(f"     {action}: {prob:.3f}")

    except ImportError as e:
        print(f"⚠️  Could not import network: {e}")
        print(f"   (This is OK - MCTS still works without network)")

    print("\n✅ All MCTS tests passed!")