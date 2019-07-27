from game import GameDing, LeftRight, FrontBack
import time
import random

if __name__ == "__main__":
    env = GameDing()
    print(env.max_diff_x, env.max_diff_y)
    time.sleep(1)
    for i in range(1000):
        dead = env.next_round(random.choice(list(LeftRight)), random.choice(list(FrontBack)))
        if dead:
            env.reset()