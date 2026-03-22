"""Debug script - trace first 50 steps of a game"""
import sys
sys.path.insert(0, '/home/user/catantron')

from simplified_reward_wrapper import SimplifiedRewardWrapper
from curriculum_trainer_v3_stable import play_opponent_turn

env = SimplifiedRewardWrapper(player_id=0, reward_mode='vp_only', victory_points_to_win=10, num_players=2)
obs, _ = env.reset()

done = False
for step in range(50):
    if done:
        break
    game = env.game_env.game
    current = game.get_current_player()
    current_id = game.players.index(current)
    phase = game.game_phase
    turn_phase = getattr(game, 'turn_phase', '?')
    
    p0_vp = game.players[0].calculate_victory_points()
    p1_vp = game.players[1].calculate_victory_points()

    if current_id == 0:
        action_mask = obs['action_mask']
        valid_actions = [i for i, m in enumerate(action_mask) if m]
        # Always pick first valid action
        action_names = ['roll_dice','place_settlement','place_road','build_settlement',
                        'build_city','build_road','buy_dev_card','end_turn','wait',
                        'trade_with_bank','do_nothing','play_knight','play_monopoly','play_year_of_plenty']
        action_id = valid_actions[0] if valid_actions else 0
        aname = action_names[action_id] if action_id < len(action_names) else str(action_id)
        print(f"[{step:3d}] P0({phase}/{turn_phase}) VP={p0_vp}/{p1_vp} mask={valid_actions} -> action={aname}")
        obs, _, terminated, truncated, info = env.step(action_id, 0, 0, trade_give_idx=0, trade_get_idx=0)
        done = terminated or truncated
    else:
        print(f"[{step:3d}] P1({phase}/{turn_phase}) VP={p0_vp}/{p1_vp} -> opponent turn")
        success = play_opponent_turn(game, current_id, 1.0, 'strong', None)
        if not success and game.can_end_turn():
            game.end_turn()
        if game.waiting_for_discards:
            env.game_env._handle_automatic_discards()
        obs = env._get_obs()
        winner = game.check_victory_conditions()
        if winner is not None:
            done = True
            print(f"  -> OPPONENT WON!")

if not done:
    print(f"\nGame not done after 50 steps (timeout)")
else:
    game = env.game_env.game
    winner = game.check_victory_conditions()
    p0_vp = game.players[0].calculate_victory_points()
    p1_vp = game.players[1].calculate_victory_points()
    print(f"\nFinal VPs: P0={p0_vp}, P1={p1_vp}, winner={game.players.index(winner) if winner else None}")
