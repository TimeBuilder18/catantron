"""
AlphaZero Training with Curriculum

Key fix: Start with WEAK opponents, gradually make them stronger.

Phases:
1. Random opponents (agent should win 50%+)
2. 50% random / 50% smart
3. 25% random / 75% smart
4. Full strength
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from game_state import GameState
from mcts import MCTS
from network_wrapper import NetworkWrapper


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def play_random_turn(env, player_id):
    """Weak opponent - makes random legal moves"""
    from game_system import ResourceType
    
    game = env.game_env.game
    player = game.players[player_id]
    
    # Initial placement
    if game.is_initial_placement_phase():
        if game.waiting_for_road:
            if game.last_settlement_vertex:
                edges = game.game_board.edges
                valid = [e for e in edges 
                        if e.structure is None and
                        (e.vertex1 == game.last_settlement_vertex or 
                         e.vertex2 == game.last_settlement_vertex)]
                if valid:
                    game.try_place_initial_road(random.choice(valid), player)
                    return True
        else:
            vertices = game.game_board.vertices
            valid = [v for v in vertices if v.structure is None and 
                    not any(adj.structure for adj in v.adjacent_vertices)]
            if valid:
                game.try_place_initial_settlement(random.choice(valid), player)
                return True
        return False
    
    # Normal play
    if game.can_roll_dice():
        game.roll_dice()
        return True
    
    if game.can_trade_or_build():
        actions = []
        res = player.resources
        
        # Check if can afford settlement (1 wood, 1 brick, 1 wheat, 1 sheep)
        can_settlement = (res[ResourceType.WOOD] >= 1 and res[ResourceType.BRICK] >= 1 and 
                         res[ResourceType.WHEAT] >= 1 and res[ResourceType.SHEEP] >= 1)
        if can_settlement:
            v = game.get_buildable_vertices_for_settlements()
            if v: actions.append(('sett', v))
        
        # Check if can afford city (2 wheat, 3 ore)
        can_city = (res[ResourceType.WHEAT] >= 2 and res[ResourceType.ORE] >= 3)
        if can_city:
            v = game.get_buildable_vertices_for_cities()
            if v: actions.append(('city', v))
        
        # Check if can afford road (1 wood, 1 brick)
        can_road = (res[ResourceType.WOOD] >= 1 and res[ResourceType.BRICK] >= 1)
        if can_road:
            e = game.get_buildable_edges()
            if e: actions.append(('road', e))
        
        # Check if can afford dev card (1 wheat, 1 sheep, 1 ore)
        can_dev = (res[ResourceType.WHEAT] >= 1 and res[ResourceType.SHEEP] >= 1 and 
                  res[ResourceType.ORE] >= 1)
        if can_dev and not game.dev_deck.is_empty():
            actions.append(('dev', None))
        
        if actions and random.random() < 0.6:
            action_type, locs = random.choice(actions)
            if action_type == 'sett':
                player.try_build_settlement(random.choice(locs))
            elif action_type == 'city':
                player.try_build_city(random.choice(locs))
            elif action_type == 'road':
                player.try_build_road(random.choice(locs))
            elif action_type == 'dev':
                game.try_buy_development_card(player)
            return True
    
    if game.can_end_turn():
        game.end_turn()
        return True
    
    return False


def play_opponent_turn(env, player_id, random_prob):
    """Play opponent with configurable randomness"""
    if random.random() < random_prob:
        return play_random_turn(env, player_id)
    else:
        from rule_based_ai import play_rule_based_turn
        return play_rule_based_turn(env, player_id)


class ReplayBuffer:
    def __init__(self, max_size=500000):
        self.buffer = deque(maxlen=max_size)
    
    def add_batch(self, examples):
        for ex in examples:
            self.buffer.append(ex)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        return {
            'observations': np.array([self.buffer[i]['observation'] for i in indices]),
            'action_probs': np.array([self.buffer[i]['action_probs'] for i in indices]),
            'values': np.array([self.buffer[i]['value'] for i in indices])
        }
    
    def __len__(self):
        return len(self.buffer)


class CurriculumMCTSTrainer:
    def __init__(self, model_path=None, num_simulations=30, learning_rate=1e-3, batch_size=None):
        self.device = get_device()
        
        if batch_size is None:
            batch_size = 2048 if self.device.type == 'cuda' else 256
        self.batch_size = batch_size
        
        self.network_wrapper = NetworkWrapper(model_path=model_path, device=str(self.device))
        self.network = self.network_wrapper.policy
        
        self.mcts = MCTS(self.network_wrapper, num_simulations=num_simulations, c_puct=1.0)
        self.num_simulations = num_simulations
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer()
        
        self.use_amp = (self.device.type == 'cuda')
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        
        self.games_played = 0
        print(f"Batch size: {self.batch_size}")
        print(f"MCTS simulations: {num_simulations}")
    
    def play_game(self, opponent_random_prob=1.0, use_mcts=True):
        """Play one game with curriculum opponent"""
        state = GameState()
        examples = []
        moves = 0
        
        while not state.is_terminal() and moves < 500:
            current = state.get_current_player()
            
            if current == 0:
                moves += 1
                obs = state.get_observation()
                
                if use_mcts:
                    action, probs_dict = self.mcts.search(state)
                    probs = self._dict_to_array(probs_dict)
                else:
                    action, probs = self._network_action(state, obs)
                
                examples.append({
                    'observation': obs['observation'].copy(),
                    'action_probs': probs
                })
                
                state.apply_action(action[0], action[1], action[2])
            else:
                # CURRICULUM: opponent strength based on random_prob
                success = play_opponent_turn(state.env, current, opponent_random_prob)
                if not success and state.env.game_env.game.can_end_turn():
                    state.env.game_env.game.end_turn()
        
        winner = state.get_winner()
        my_vp = state.get_victory_points(0)
        
        # VALUE BASED ON VP, NOT JUST WIN/LOSE
        # This gives gradient signal even when losing!
        # 10 VP (win) = 1.0, 5 VP = 0.5, 2 VP = 0.2
        value = my_vp / 10.0
        
        # Bonus for actually winning
        if winner == 0:
            value = 1.0
        
        training_examples = [
            {'observation': ex['observation'], 'action_probs': ex['action_probs'], 'value': value}
            for ex in examples
        ]
        self.replay_buffer.add_batch(training_examples)
        self.games_played += 1
        
        return winner, my_vp
    
    def _dict_to_array(self, probs_dict):
        probs = np.zeros(11, dtype=np.float32)
        for (action_id, _, _), p in probs_dict.items():
            probs[action_id] += p
        if probs.sum() > 0:
            probs /= probs.sum()
        return probs
    
    def _network_action(self, state, obs):
        """Fast action without MCTS"""
        with torch.no_grad():
            observation = torch.FloatTensor(obs['observation']).unsqueeze(0).to(self.device)
            action_mask = torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(self.device)
            vertex_mask = torch.FloatTensor(obs['vertex_mask']).unsqueeze(0).to(self.device)
            edge_mask = torch.FloatTensor(obs['edge_mask']).unsqueeze(0).to(self.device)
            
            action_probs, vertex_probs, edge_probs, _, _, _ = self.network.forward(
                observation, action_mask, vertex_mask, edge_mask
            )
            
            ap = action_probs.cpu().numpy()[0]
            vp = vertex_probs.cpu().numpy()[0]
            ep = edge_probs.cpu().numpy()[0]
        
        legal = state.get_legal_actions()
        if not legal:
            return (7, 0, 0), np.zeros(11)
        
        weights = {}
        for (a, v, e) in legal:
            w = ap[a]
            if a in [1, 3, 4]: w *= vp[v]
            elif a in [2, 5]: w *= ep[e]
            weights[(a, v, e)] = w
        
        actions = list(weights.keys())
        w = np.array(list(weights.values()))
        w = w / (w.sum() + 1e-8)
        w = 0.8 * w + 0.2 / len(w)
        w = w / w.sum()
        
        idx = np.random.choice(len(actions), p=w)
        return actions[idx], ap
    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        batch = self.replay_buffer.sample(self.batch_size)
        
        obs = torch.FloatTensor(batch['observations']).to(self.device)
        target_probs = torch.FloatTensor(batch['action_probs']).to(self.device)
        target_values = torch.FloatTensor(batch['values']).to(self.device)
        
        self.network.train()
        
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                action_probs, _, _, _, _, value = self.network.forward(obs)
                value = value.squeeze()
                policy_loss = -torch.sum(target_probs * torch.log(action_probs + 1e-8), dim=1).mean()
                value_loss = F.mse_loss(value, target_values)
                loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            action_probs, _, _, _, _, value = self.network.forward(obs)
            value = value.squeeze()
            policy_loss = -torch.sum(target_probs * torch.log(action_probs + 1e-8), dim=1).mean()
            value_loss = F.mse_loss(value, target_values)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()
        
        return {'policy': policy_loss.item(), 'value': value_loss.item()}
    
    def train(self, 
              games_per_phase=200,
              parallel_games=8, 
              use_mcts=False,  # Start without MCTS for speed
              mcts_sims=10,
              save_path='models/curriculum_mcts'):
        """
        Curriculum training with phases
        """
        os.makedirs('models', exist_ok=True)
        
        # Curriculum phases: (random_prob, description)
        phases = [
            (1.0, "Random opponents"),
            (0.75, "75% random"),
            (0.5, "50% random"),
            (0.25, "25% random"),
            (0.0, "Full strength"),
        ]
        
        print("\n" + "=" * 70)
        print("CURRICULUM MCTS TRAINING")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Games per phase: {games_per_phase}")
        print(f"Parallel games: {parallel_games}")
        print(f"Use MCTS: {use_mcts} (sims={mcts_sims})")
        print("=" * 70 + "\n")
        
        total_wins = 0
        total_games = 0
        start_time = time.time()
        
        for phase_idx, (random_prob, phase_name) in enumerate(phases):
            print(f"\n{'='*70}")
            print(f"PHASE {phase_idx + 1}: {phase_name}")
            print(f"{'='*70}")
            
            phase_wins = 0
            phase_vp = 0
            recent_wins = deque(maxlen=50)
            
            game_num = 0
            while game_num < games_per_phase:
                batch_start = time.time()
                
                # Play games in parallel
                games_to_play = min(parallel_games, games_per_phase - game_num)
                
                with ThreadPoolExecutor(max_workers=games_to_play) as executor:
                    futures = [
                        executor.submit(self.play_game, random_prob, use_mcts)
                        for _ in range(games_to_play)
                    ]
                    
                    for future in as_completed(futures):
                        winner, vp = future.result()
                        game_num += 1
                        total_games += 1
                        phase_vp += vp
                        
                        if winner == 0:
                            phase_wins += 1
                            total_wins += 1
                            recent_wins.append(1)
                        else:
                            recent_wins.append(0)
                
                batch_time = time.time() - batch_start
                elapsed = time.time() - start_time
                
                # Progress
                recent_wr = sum(recent_wins) / len(recent_wins) * 100 if recent_wins else 0
                avg_vp = phase_vp / game_num if game_num > 0 else 0
                speed = total_games / elapsed * 60
                
                print(f"  Game {game_num:3d}/{games_per_phase} | "
                      f"WR: {recent_wr:5.1f}% | "
                      f"VP: {avg_vp:.1f} | "
                      f"Speed: {speed:.1f} g/min")
                
                # Train
                if len(self.replay_buffer) >= self.batch_size:
                    losses = [self.train_step() for _ in range(10)]
                    losses = [l for l in losses if l]
                    if losses:
                        avg_p = np.mean([l['policy'] for l in losses])
                        avg_v = np.mean([l['value'] for l in losses])
                        print(f"    └─ Train: policy={avg_p:.4f}, value={avg_v:.4f}")
            
            # Phase summary
            phase_wr = phase_wins / games_per_phase * 100
            phase_avg_vp = phase_vp / games_per_phase
            print(f"\n  Phase {phase_idx + 1} Complete: WR={phase_wr:.1f}%, Avg VP={phase_avg_vp:.1f}")
            
            # Save after each phase
            self.save(f"{save_path}_phase{phase_idx + 1}.pt")
            
            # If not winning enough, don't advance (optional)
            if phase_wr < 30 and phase_idx < len(phases) - 1:
                print(f"  ⚠️ Win rate low, replaying phase...")
                # Could add logic to replay phase
        
        # Final save
        self.save(f"{save_path}_final.pt")
        
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print(f"Total games: {total_games}")
        print(f"Overall win rate: {total_wins/total_games*100:.1f}%")
        print(f"Time: {total_time/60:.1f} min")
        print("=" * 70)
    
    def save(self, path):
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'games_played': self.games_played,
        }, path)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--games-per-phase', type=int, default=200)
    parser.add_argument('--parallel', type=int, default=8)
    parser.add_argument('--mcts', action='store_true', help='Use MCTS (slower but smarter)')
    parser.add_argument('--sims', type=int, default=10, help='MCTS simulations')
    args = parser.parse_args()
    
    trainer = CurriculumMCTSTrainer(num_simulations=args.sims)
    trainer.train(
        games_per_phase=args.games_per_phase,
        parallel_games=args.parallel,
        use_mcts=args.mcts,
        mcts_sims=args.sims
    )
