import argparse
from collections import deque

import gym
import numpy as np
import sympy
import tensorflow as tf

from tqdm import trange
from train_QML import interact_env
import wandb
import os


def get_parsed(text=None):
    parser = argparse.ArgumentParser(description="Define and run the experiment with one config")
    parser.add_argument('--run', type=int, default=1, help="")
    parser.add_argument('--model', type=str, default="Small", help="")
    if text is not None:
        arguments = vars(parser.parse_args(text))
    else:
        arguments = vars(parser.parse_args())

    return arguments


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


def main():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    arguments = get_parsed()


    project = "CartPole-V1"
    if arguments['model'] == "Small":
        layers = [9, 4]
    elif arguments['model'] == "Deep":
        layers = [32, 32]
    elif arguments['model'] == "Shallow":
        layers = [64]
    elif arguments['model'] == "Shallow_small":
        layers = [13]
    name = "run-" + str(arguments["run"])
    arg_mod = "Classical_" + arguments['model'] + "_v2"
    run = wandb.init(project=project,
                     group=arg_mod,
                     name=arg_mod + "_" + name,
                     config=arguments)
    save_dir = os.path.join(project, arg_mod, name)

    n_state = 4  # Dimension of the state vectors in CartPole
    n_actions = 2  # Number of actions in CartPole

    model = generate_model(layers, n_state, n_actions)
    model_target = generate_model(layers, n_state, n_actions)
    model_target.set_weights(model.get_weights())
    print(model_target.summary())

    gamma = 0.99
    n_episodes = 2000

    # Define replay memory
    max_memory_length = 10000  # Maximum replay length
    replay_memory = deque(maxlen=max_memory_length)

    epsilon = 1.0  # Epsilon greedy parameter
    epsilon_min = 0.01  # Minimum epsilon greedy parameter
    decay_epsilon = 0.95  # Decay rate of epsilon greedy parameter
    batch_size = 32
    steps_per_update = 10  # Train the model every x steps
    steps_per_target_update = 30  # Update the target model every x steps

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    env = gym.make("CartPole-v1")

    episode_reward_history = []
    step_count = 0
    avg_rewards_100= -100
    t = trange(n_episodes, desc='Training', leave=True)

    for episode in t:
        episode_reward = 0
        state = env.reset()
        for step in range(200):
            # Interact with env
            interaction = interact_env(state, model, epsilon, n_actions, env)

            # Store interaction in the replay memory
            replay_memory.append(interaction)

            state = interaction['next_state']
            episode_reward += interaction['reward']
            step_count += 1

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

        # Decay epsilon
        epsilon = max(epsilon * decay_epsilon, epsilon_min)
        episode_reward_history.append(episode_reward)
        wandb.log({'episode_reward': episode_reward}, step=step_count)

        if (episode + 1) % 10 == 0:
            avg_rewards = np.mean(episode_reward_history[-10:])
            wandb.log({'episode_reward_avg_last_10': avg_rewards, 'episode': episode + 1}, step=step_count)
            t.set_description("Episode {}/{}, avg 10 rew {}, avg 100 rew {}".format(episode + 1, n_episodes, avg_rewards, avg_rewards_100))
            t.refresh()  # to show immediately the update

        if (episode + 1) % 100 == 0:
            avg_rewards_100 = np.mean(episode_reward_history[-100:])
            wandb.log({'episode_reward_avg_last_100': avg_rewards_100, 'episode': episode + 1}, step=step_count)
            t.set_description("Episode {}/{}, avg 10 rew {}, avg 100 rew {}".format(episode + 1, n_episodes, avg_rewards, avg_rewards_100))
            t.refresh()  # to show immediately the update
            if avg_rewards_100 >= 195.0:  # 500.0
                break
    try:
        os.makedirs(save_dir)
    except OSError:
        print(" %s already exists or error" % save_dir)
    else:
        print("Successfully created the directory %s " % save_dir)

    model_target.save_weights(save_dir + "/")




    run.finish()


if __name__ == "__main__":
    main()
