import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import durakk_env
import numpy as np

env = durakk_env.DurakEnv()
ok_games = 0
for game in range(10):
    env.reset()
    moves = 0
    while not env.is_game_over() and moves < 500:
        probs = env.run_ismcts(0.05, 1)
        mask = np.array(env.get_legal_action_mask())
        p = np.array(probs) * mask
        action = int(np.argmax(p)) if p.sum() > 0 else int(np.argmax(mask))
        env.step(action)
        moves += 1
    if env.is_game_over():
        ok_games += 1
print(f"Доиграли до конца: {ok_games}/10")
