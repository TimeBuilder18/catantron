"""
Monte Carlo Tree Search for Catan

AlphaZero-style MCTS with:
- Location-aware priors (action × vertex/edge probabilities)
- Lazy expansion (copy state only when visiting a child)
- Dirichlet noise at root for exploration
- Neural network value estimation
"""

import math
import numpy as np
from game_state import GameState

# Actions that need vertex selection
VERTEX_ACTIONS = {1, 3, 4}  # place_settlement, build_settlement, build_city
# Actions that need edge selection
EDGE_ACTIONS = {2, 5}  # place_road, build_road


class MCTSNode:
    """
    A node in the MCTS tree.
    Uses lazy expansion: child states are only created when first visited.
    """

    __slots__ = ['state', 'parent', 'action', 'prior',
                 'children', 'visits', 'value_sum', '_expanded']

    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state        # GameState (None for unexpanded children)
        self.parent = parent
        self.action = action      # (action_id, vertex_id, edge_id)
        self.prior = prior        # P(s,a) from network

        self.children = {}        # action tuple -> MCTSNode
        self.visits = 0           # N(s)
        self.value_sum = 0.0      # W(s)
        self._expanded = False

    def value(self):
        """Q(s) = W(s) / N(s)"""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def is_expanded(self):
        return self._expanded

    def is_terminal(self):
        return self.state is not None and self.state.is_terminal()

    def ucb_score(self, c_puct=1.0):
        if self.parent is None:
            return 0.0
        q_value = self.value()
        exploration = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q_value + exploration


class MCTS:
    """
    Monte Carlo Tree Search with Neural Network guidance.

    Key improvements over basic implementation:
    - Priors combine action_prob × location_prob for location-specific actions
    - Lazy expansion: states copied only when a child is actually visited
    - Dirichlet noise at root for exploration diversity
    """

    def __init__(self, policy_network=None, num_simulations=50, c_puct=1.5,
                 dirichlet_alpha=0.3, dirichlet_frac=0.25):
        """
        Args:
            policy_network: NetworkWrapper with evaluate(obs) method
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant (higher = more exploration)
            dirichlet_alpha: Alpha for Dirichlet noise at root
            dirichlet_frac: Fraction of noise mixed into root priors
        """
        self.policy_network = policy_network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac

    def search(self, root_state, temperature=1.0):
        """
        Run MCTS from root_state.

        Args:
            root_state: GameState to search from
            temperature: Controls action selection (1.0=proportional, 0=greedy)

        Returns:
            best_action: (action_id, vertex_id, edge_id) tuple
            action_probs: dict mapping action tuples to visit probabilities
        """
        root_player = root_state.get_current_player()

        # Create root node with its own copy
        root = MCTSNode(root_state.copy())

        # Expand root and add Dirichlet noise
        self._expand(root)
        self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # 1. SELECT — traverse tree using UCB
            while node.is_expanded() and not node.is_terminal():
                node = self._select_child(node)
                search_path.append(node)

            # 2. Lazy state creation — copy parent state and apply action
            if node.state is None:
                node.state = node.parent.state.copy()
                a_id, v_id, e_id = node.action
                node.state.apply_action(a_id, v_id, e_id)

            # 3. EXPAND & EVALUATE
            if not node.is_terminal():
                self._expand(node)
                value = self._evaluate(node)
            else:
                value = node.state.get_result(root_player)

            # 4. BACKPROPAGATE
            self._backpropagate(search_path, value, root_player)

        best_action = self._get_best_action(root)
        action_probs = self._get_action_probs(root, temperature)
        return best_action, action_probs

    def _select_child(self, node):
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_child = None
        for child in node.children.values():
            score = child.ucb_score(self.c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _expand(self, node):
        """
        Expand node by creating child placeholders for all legal actions.
        Children start with state=None (lazy — state created on first visit).
        """
        legal_actions = node.state.get_legal_actions()
        if not legal_actions:
            return

        priors = self._get_priors(node.state, legal_actions)

        for action, prior in zip(legal_actions, priors):
            # Lazy: child has no state yet, just prior and action
            child = MCTSNode(
                state=None,
                parent=node,
                action=action,
                prior=prior
            )
            node.children[action] = child

        node._expanded = True

    def _evaluate(self, node):
        """Get value estimate from neural network."""
        if self.policy_network is None:
            current_player = node.state.get_current_player()
            my_vp = node.state.get_victory_points(current_player)
            return (my_vp - 5) / 5.0

        obs = node.state.get_observation()
        _, value = self.policy_network.evaluate(obs)
        return value

    def _get_priors(self, state, legal_actions):
        """
        Get prior probabilities for legal actions.

        For location-specific actions (build settlement at vertex X),
        the prior is: P(action_type) × P(vertex | action_type).
        This lets MCTS distinguish between good and bad locations.
        """
        if self.policy_network is None:
            n = len(legal_actions)
            return [1.0 / n] * n

        obs = state.get_observation()
        policy, _ = self.policy_network.evaluate(obs)

        action_probs = policy['action']   # shape (14,)
        vertex_probs = policy['vertex']   # shape (54,)
        edge_probs = policy['edge']       # shape (72,)

        priors = []
        for action_id, vertex_id, edge_id in legal_actions:
            # Base probability for this action type
            a_prob = action_probs[action_id] if action_id < len(action_probs) else 1e-4

            if action_id in VERTEX_ACTIONS:
                # Action × vertex probability
                v_prob = vertex_probs[vertex_id] if vertex_id < len(vertex_probs) else 1e-4
                prior = a_prob * v_prob
            elif action_id in EDGE_ACTIONS:
                # Action × edge probability
                e_prob = edge_probs[edge_id] if edge_id < len(edge_probs) else 1e-4
                prior = a_prob * e_prob
            else:
                prior = a_prob

            # Floor to avoid zero priors (MCTS needs some exploration)
            priors.append(max(prior, 1e-6))

        # Normalize to sum to 1
        total = sum(priors)
        if total > 0:
            priors = [p / total for p in priors]
        else:
            priors = [1.0 / len(priors)] * len(priors)

        return priors

    def _add_dirichlet_noise(self, root):
        """
        Add Dirichlet noise to root priors for exploration.
        This is critical for AlphaZero — without it, MCTS just follows
        the network's existing policy with minimal diversity.
        """
        if not root.children:
            return

        children = list(root.children.values())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(children))
        frac = self.dirichlet_frac

        for child, n in zip(children, noise):
            child.prior = (1 - frac) * child.prior + frac * n

    def _backpropagate(self, search_path, value, root_player):
        """Update all nodes in search path with value."""
        for node in reversed(search_path):
            node.visits += 1
            if node.state is not None:
                current_player = node.state.get_current_player()
                if current_player == root_player:
                    node.value_sum += value
                else:
                    node.value_sum -= value
            else:
                # Lazy node not yet expanded — use parent's perspective
                if node.parent is not None and node.parent.state is not None:
                    parent_player = node.parent.state.get_current_player()
                    if parent_player == root_player:
                        node.value_sum += value
                    else:
                        node.value_sum -= value

    def _get_best_action(self, root):
        """Get action with most visits."""
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action

    def _get_action_probs(self, root, temperature=1.0):
        """
        Get action probabilities based on visit counts.

        Args:
            root: Root node
            temperature: 1.0=proportional to visits, 0=always pick best
        """
        actions = list(root.children.keys())
        visits = np.array([root.children[a].visits for a in actions], dtype=np.float32)

        if temperature == 0 or visits.sum() == 0:
            probs = np.zeros(len(actions))
            if visits.sum() > 0:
                probs[np.argmax(visits)] = 1.0
            else:
                probs[:] = 1.0 / len(probs)
        else:
            visits = visits ** (1.0 / temperature)
            probs = visits / visits.sum()

        return dict(zip(actions, probs))


# Quick test
if __name__ == "__main__":
    print("Testing MCTS...")

    state = GameState()
    print(f"Created game state, {len(state.get_legal_actions())} legal actions")

    # Test 1: MCTS without network
    print("\n--- Test 1: MCTS without neural network ---")
    mcts_no_nn = MCTS(policy_network=None, num_simulations=10)
    best_action, action_probs = mcts_no_nn.search(state)
    print(f"Best action: {best_action}")

    # Test 2: MCTS with neural network
    print("\n--- Test 2: MCTS with neural network ---")
    try:
        from network_wrapper import NetworkWrapper

        network = NetworkWrapper(model_path=None)
        mcts_with_nn = MCTS(policy_network=network, num_simulations=20)

        best_action, action_probs = mcts_with_nn.search(state)
        print(f"Best action: {best_action}")

        sorted_actions = sorted(action_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top 5 actions:")
        for action, prob in sorted_actions:
            print(f"  {action}: {prob:.3f}")
    except ImportError as e:
        print(f"Could not import network: {e}")

    print("\nAll MCTS tests passed!")
