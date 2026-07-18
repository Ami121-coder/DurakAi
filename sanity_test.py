import sys, os
import numpy as np
# Добавим пути как в training/train.py
sys.path.append(os.path.join(os.path.dirname(__file__), "engine", "build", "Release"))
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

try:
    import durakk_env
    print("durakk_env successfully imported!")
except ImportError as e:
    print("Failed to import durakk_env:", e)
    sys.exit(1)

env = durakk_env.DurakEnv()
print("ACTION_SIZE is:", durakk_env.ACTION_SIZE)

wins = 0
for i in range(100):
    env.reset_with_seed(i)
    moves = 0
    while not env.is_game_over() and moves < 500:
        mask = env.get_legal_action_mask()
        valid_actions = np.where(np.array(mask) > 0)[0]
        if len(valid_actions) > 0:
            env.step(int(valid_actions[0]))
        else:
            break
        moves += 1
    if env.is_game_over():
        wins += 1

print(f"Sanity test: played {wins} games to completion out of 100.")
