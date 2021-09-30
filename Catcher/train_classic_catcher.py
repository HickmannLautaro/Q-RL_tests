import os
import sys
import time
from collections import deque

import cirq
import sympy
import tensorflow as tf
from tqdm import trange

sys.path.insert(0, '../PyGame-Learning-Environment/')
from ple.games.catcher_discrete import Catcher
from ple import PLE
import numpy as np
import matplotlib.pyplot as plt

from train_QML_catcher import interact_env
from PIL import Image


@tf.function
def Q_learning_update(states, actions, rewards, next_states, done, model, model_target, gamma, optimizer, n_actions):
    states = tf.convert_to_tensor(states)
    actions = tf.convert_to_tensor(actions)
    rewards = tf.convert_to_tensor(rewards)
    next_states = tf.convert_to_tensor(next_states)
    done = tf.convert_to_tensor(done)

    # Compute their target q_values and the masks on sampled actions
    future_rewards = model_target([next_states])
    target_q_values = rewards + (gamma * tf.reduce_max(future_rewards, axis=1)
                                 * (1.0 - done))
    masks = tf.one_hot(actions, n_actions)

    # Train the model on the states and target Q-values
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        q_values = model([states])
        q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        loss = tf.keras.losses.mean_squared_error(target_q_values, q_values_masked)  # With huber it completly breaks.

    # Backpropagation
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def generate_model(layers, n_state, n_actions):
    if len(layers) > 1:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(layers[0], activation="elu", input_shape=[n_state]),
            tf.keras.layers.Dense(layers[1], activation="elu"),
            tf.keras.layers.Dense(n_actions)
        ])
    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(layers[0], activation="elu", input_shape=[n_state]),
            tf.keras.layers.Dense(n_actions)
        ])
    return model


def process_state(state):
    return np.array([state.values()])

def show_epopch(env, model, epsilon,n_actions, steps_target_per_episode,episode, save = False):
    actions = [97, None, 100]

    env.reset_game()
    print("\n")
    img = Image.new("RGB", (512, 512), (120, 120, 120))
    frames = []
    frames.append(img)
    frames.append(img)
    frames.append(img)

    for step in range(steps_target_per_episode+1):

        state = np.array(list(env.getGameState()[0]))
        if save:

            observation = Image.fromarray(env.getScreenRGB()).rotate(-90)
            frames.append(observation)

        state_array = np.array(state)

        state = tf.convert_to_tensor([state_array])


        # Sample action
        coin = np.random.random()
        if coin > epsilon:
            q_vals = model([state])
            action = int(tf.argmax(q_vals[0]).numpy())
        else:
            action = np.random.choice(n_actions)


        action_names = ["left","stay", "right"]
        reward = env.act(actions[action])

        print(f"player x position {state_array[0]:3d},   fruits x position {state_array[1]:3d},   fruits y position {state_array[2]:3d}, action {action_names[action]}, reward {reward}, score {env.score()} ")

        if env.game_over():
            break
    print("Score: ", env.score(),"\n")
    if save:
        frames[1].save(f"./Saves/Simplified_catcher_epsiode_{episode}_steps_{step}.gif", save_all=True, append_images=frames[2:], optimize=False, duration=step*10, loop=0)


def main():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    layers = [32,32]  # layers = [9, 4] layers = [64] layers = [13] layers = [32, 32]

    name = "Run-1"
    save_dir = os.path.join("Saves", name)

    n_state = 3  # Dimension of the state vectors in CartPole
    n_actions = 3  # Number of actions in CartPole

    model = generate_model(layers, n_state, n_actions)
    model_target = generate_model(layers, n_state, n_actions)
    model_target.set_weights(model.get_weights())
    print(model_target.summary())

    gamma = 0.99
    n_episodes = 50000
    steps_target_per_episode =250
    # Define replay memory
    max_memory_length = 10000  # Maximum replay length
    replay_memory = deque(maxlen=max_memory_length)

    epsilon = 1.0  # Epsilon greedy parameter
    epsilon_min = 0.01  # Minimum epsilon greedy parameter
    decay_epsilon = 0.95  # Decay rate of epsilon greedy parameter
    batch_size = 32
    steps_per_update = 25  # Train the model every x steps
    steps_per_target_update = 75  # Update the target model every x steps

    optimizer = tf.keras.optimizers.Adam()#learning_rate=1e-2)

    game = Catcher(width=512, height=512, init_lives=3)
    p = PLE(game, display_screen=False, state_preprocessor=process_state)#,  force_fps=False, fps=240)
    env = PLE(game, display_screen=True, state_preprocessor=process_state, force_fps=False, fps=2)

    episode_reward_history = []
    episode_score_history = []

    steps_per_ep_history = []

    step_count = 0
    t = trange(n_episodes, desc='Training', leave=True)

    for episode in t:
        episode_reward = 0
        p.reset_game()
        state = np.array(list(p.getGameState()[0]))
        steps_in_episode = 0

        for step in range(steps_target_per_episode+1):
            # Interact with env
            interaction = interact_env(state, model, epsilon, n_actions, p)

            # Store interaction in the replay memory
            replay_memory.append(interaction)

            state = interaction['next_state']
            episode_reward += interaction['reward']
            step_count += 1
            steps_in_episode += 1

            # Update model
            if step_count % steps_per_update == 0:
                # Sample a batch of interactions and update Q_function
                training_batch = np.random.choice(replay_memory, size=batch_size)
                Q_learning_update(np.asarray([x['state'] for x in training_batch]),
                                  np.asarray([x['action'] for x in training_batch]),
                                  np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
                                  np.asarray([x['next_state'] for x in training_batch]),
                                  np.asarray([x['done'] for x in training_batch], dtype=np.float32),
                                  model, model_target, gamma, optimizer, n_actions)
            # Update target model
            if step_count % steps_per_target_update == 0:
                model_target.set_weights(model.get_weights())


            # Check if the episode is finished
            if interaction['done']:
                break
        if episode % 1000 ==0:
            if episode ==0:
                show_epopch(env, model_target, epsilon, n_actions, steps_target_per_episode,episode, True)
            else:
                show_epopch(env, model_target, epsilon, n_actions, steps_target_per_episode,episode)
        # Decay epsilon
        epsilon = max(epsilon * decay_epsilon, epsilon_min)
        episode_reward_history.append(episode_reward)
        episode_score_history.append(p.score())
        steps_per_ep_history.append(steps_in_episode)
        avg_rewards = np.mean(episode_reward_history[-100:])

        avg_rewards_200 = np.mean(episode_reward_history[-200:])
        t.set_description("Episode {:5d}/{:5d}, steps in ep {:4d}, score {:04.2f}, avg 100 rew {:04.2f}, avg 200 rew {:04.2f}".format(episode + 1, n_episodes, steps_in_episode, p.score(), avg_rewards, avg_rewards_200))
        t.refresh()  # to show immediately the update
        if avg_rewards_200 > (steps_target_per_episode//5 )*.95:
            break

    plt.plot(episode_score_history)
    plt.title("Score per trial")
    plt.show()
    plt.plot(episode_reward_history)
    plt.title("Reward per trial")
    plt.show()
    plt.title("Steps per trial")
    plt.plot(steps_per_ep_history)
    plt.show()
    show_epopch(env, model_target, epsilon, n_actions, steps_target_per_episode,episode, True)
    try:
        os.makedirs(save_dir)
    except OSError:
        print(" %s already exists or error" % save_dir)
    else:
        print("Successfully created the directory %s " % save_dir)

    model_target.save_weights(save_dir + "/target_weights")


if __name__ == "__main__":
    main()
