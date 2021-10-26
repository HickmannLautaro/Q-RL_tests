import sys


sys.path.insert(0, '../PyGame-Learning-Environment/')
from ple.games.avoider_v_1 import Avoider_v_1
from ple import PLE
import numpy as np

from PIL import Image
import tensorflow as tf



def show_epopch(env, n_actions, steps_target_per_episode, save_dir):
    actions = [97, None, 100]

    env.reset_game()
    print("\n")
    img = Image.new("RGB", (512, 512), (120, 120, 120))
    frames = [img, img, img]

    for step in range(steps_target_per_episode + 1):

        state = np.array(list(env.getGameState()[0]))

        observation = Image.fromarray(env.getScreenRGB()).rotate(-90)
        frames.append(observation)

        state_array = np.array(state)

        state = tf.convert_to_tensor([state_array])

        # Sample action
        coin = np.random.random()

        action = np.random.choice(n_actions)
        action = input("action: ")

        try:
            action = int(action)
        except:
            action = 1


        action_names = ["left", "stay", "right"]
        reward = env.act(actions[action])

        # print(f"player x position {state_array[0]:3d},   fruits x position {state_array[1]:3d},   fruits y position {state_array[2]:3d}, action {action_names[action]}, reward {reward}, score {env.score()} ")
        # print(f"player x position {int(state_array[0])}, line 1 y position {state_array[1]:.2f}, line 2 y position {state_array[2]:.2f}, line 3 y position {state_array[3]:.2f}, action {action_names[action]}, reward {reward}, score {env.score()} ")
        print(f"player x position {int(state_array[0])}, "
              f"line 1 y position {state_array[1]:.2f}, line 1 vel {state_array[2]:.2f}, "
              f"line 2 y position {state_array[3]:.2f}, line 2 vel {state_array[4]:.2f},"
              f"line 3 y position {state_array[5]:.2f}, line 3 vel {state_array[6]:.2f}, action {action_names[action]}, reward {reward}, score {env.score()}")


        if env.game_over():
            break
    print("Score: ", env.score(), "\n")

    frames[1].save(f"{save_dir}_demo.gif", save_all=True, append_images=frames[2:], optimize=False, loop=0)



def process_state(state):
    return np.array([state.values()])


def main():

    # run inference on every image, assuming my_model returns the
    # predicted label, and the ground truth labels are available


    n_state = 3  # Dimension of the state vectors in CartPole
    n_actions = 3  # Number of actions in CartPole
    steps_target_per_episode = 1000



    save_dir = "./avoider_phase_1_v_1"
    game = Avoider_v_1(width=512, height=512, init_lives=1)
    env = PLE(game, display_screen=True, state_preprocessor=process_state)  # , force_fps=False, fps=2)
    show_epopch(env, n_actions, steps_target_per_episode, save_dir)



if __name__ == "__main__":
    main()
