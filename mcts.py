"""
Monte Carlo Tree Search for Catan

AlphaZero-style MCTS with:
- Location-aware priors (action × vertex/edge probabilities)
- Lazy expansion (copy state only when visiting a child)
- Dirichlet noise at root for exploration
- Neural network value estimation
- Single NN call per simulation (combined priors + value)
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

    Key improvements:
    - Single NN call per simulation (combined expand + evaluate)
    - Priors combine action_prob × location_prob for location-specific actions
    - Lazy expansion: states copied only when a child is actually visited
    - Dirichlet noise at root for exploration diversity
    - Skips search entirely for forced moves (1 legal action)
    """

    def __init__(self, policy_network=None, num_simulations=50, c_puct=1.5,
                 dirichlet_alpha=0.3, dirichlet_frac=0.25):
        self.policy_network = policy_network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        # Optional callable for batched inference: fn(obs) -> (policy_dict, value)
        self._infer_fn = None

    def set_infer_fn(self, fn):
        """Set a batched inference function: fn(obs_dict) -> (policy_dict, value).
        This allows MCTS to route NN calls through CentralInferenceServer."""
        self._infer_fn = fn

    def search(self, root_state, temperature=1.0):
        """
        Run MCTS from root_state.

        Returns:
            best_action: (action_id, vertex_id, edge_id) tuple
            action_probs: dict mapping action tuples to visit probabilities
        """
        root_player = root_state.get_current_player()

        # Create root node with its own copy
        root = MCTSNode(root_state.copy())

        # Expand root: single NN call returns both priors and value
        root_value = self._expand_and_evaluate(root)

        # Skip search for forced moves (only 1 legal action)
        if len(root.children) <= 1:
            if root.children:
                action = next(iter(root.children))
                return action, {action: 1.0}
            else:
                return (0, 0, 0), {(0, 0, 0): 1.0}

        # Add Dirichlet noise at root
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

            # 3. EXPAND & EVALUATE in single NN call
            if not node.is_terminal():
                value = self._expand_and_evaluate(node)
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

    def _expand_and_evaluate(self, node):
        """
        Expand node AND get value estimate in a SINGLE neural network call.

        Previously _expand and _evaluate were separate, requiring 2 NN calls
        per simulation. Now we do 1 call and use the result for both.

        Returns:
            value: float in [-1, 1]
        """
        legal_actions = node.state.get_legal_actions()
        if not legal_actions:
            node._expanded = True
            return 0.0

        # Single NN call — get both policy (for priors) and value
        obs = node.state.get_observation()
        has_network = self.policy_network is not None or self._infer_fn is not None

        if has_network:
            if self._infer_fn is not None:
                policy, value = self._infer_fn(obs)
            else:
                policy, value = self.policy_network.evaluate(obs)

            # Compute priors from policy
            action_probs = policy['action']
            vertex_probs = policy['vertex']
            edge_probs = policy['edge']

            priors = []
            for action_id, vertex_id, edge_id in legal_actions:
                a_prob = action_probs[action_id] if action_id < len(action_probs) else 1e-4

                if action_id in VERTEX_ACTIONS:
                    v_prob = vertex_probs[vertex_id] if vertex_id < len(vertex_probs) else 1e-4
                    prior = a_prob * v_prob
                elif action_id in EDGE_ACTIONS:
                    e_prob = edge_probs[edge_id] if edge_id < len(edge_probs) else 1e-4
                    prior = a_prob * e_prob
                else:
                    prior = a_prob

                priors.append(max(prior, 1e-6))

            # Normalize
            total = sum(priors)
            if total > 0:
                priors = [p / total for p in priors]
            else:
                priors = [1.0 / len(priors)] * len(priors)
        else:
            # No network — uniform priors, heuristic value
            n = len(legal_actions)
            priors = [1.0 / n] * n
            current_player = node.state.get_current_player()
            my_vp = node.state.get_victory_points(current_player)
            value = (my_vp - 5) / 5.0

        # Create children with computed priors
        for action, prior in zip(legal_actions, priors):
            child = MCTSNode(state=None, parent=node, action=action, prior=prior)
            node.children[action] = child

        node._expanded = True
        return value

    def _add_dirichlet_noise(self, root):
        """Add Dirichlet noise to root priors for exploration."""
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
        """Get action probabilities based on visit counts."""
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
