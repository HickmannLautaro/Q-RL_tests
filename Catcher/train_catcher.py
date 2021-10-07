import argparse
import os
import sys
from collections import deque
from datetime import timedelta
from random import randint
from time import sleep

import cirq
import sympy
import tensorflow as tf
from tqdm import trange

sys.path.insert(0, '../PyGame-Learning-Environment/')
from ple.games.catcher_discrete import Catcher
from ple import PLE
import numpy as np

from PIL import Image
import wandb


def interact_env(state, model, epsilon, n_actions, p):
    # Preprocess state
    state_array = np.array(state)
    state = tf.convert_to_tensor([state_array])

    # Sample action
    coin = np.random.random()
    if coin > epsilon:
        q_vals = model([state])
        action = int(tf.argmax(q_vals[0]).numpy())
    else:
        action = np.random.choice(n_actions)

    actions = [97, None, 100]
    # Apply sampled action in the environment, receive reward and next state
    reward = p.act(actions[action])
    next_state = np.array(list(p.getGameState()[0]))
    done = p.game_over()

    interaction = {'state': state_array, 'action': action, 'next_state': next_state.copy(),
                   'reward': reward, 'done': float(done)}

    return interaction


@tf.function
def Q_learning_update_classic(states, actions, rewards, next_states, done, model, model_target, gamma, optimizer, n_actions):
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
        loss = tf.keras.losses.mean_squared_error(target_q_values, q_values_masked)

    # Backpropagation
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return grads


@tf.function
def Q_learning_update_quantum(states, actions, rewards, next_states, done, model, model_target, gamma, n_actions, optimizer_in, optimizer_var, optimizer_out, w_in, w_var, w_out):
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

        loss = tf.keras.losses.Huber()(target_q_values, q_values_masked)  # Original
        #loss = tf.keras.losses.mean_squared_error(target_q_values, q_values_masked)

        # Backpropagation
    grads = tape.gradient(loss, model.trainable_variables)

    for optimizer, w in zip([optimizer_in, optimizer_var, optimizer_out], [w_in, w_var, w_out]):
        optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])
    return grads


def generate_circuit(qubits, n_layers):
    """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
    # Number of qubits
    n_qubits = len(qubits)

    # Sympy symbols for variational angles
    params = sympy.symbols(f'theta(0:{3 * (n_layers + 1) * n_qubits})')
    params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))

    # Sympy symbols for encoding angles
    inputs = sympy.symbols(f'x(0:{n_qubits})' + f'(0:{n_layers})')
    inputs = np.asarray(inputs).reshape((n_layers, n_qubits))

    # Define circuit
    circuit = cirq.Circuit()
    for l in range(n_layers):
        # Variational layer
        circuit += cirq.Circuit(one_qubit_rotation(q, params[l, i]) for i, q in enumerate(qubits))
        circuit += entangling_layer(qubits)
        # Encoding layer
        circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(qubits))

    # Last varitional layer
    circuit += cirq.Circuit(one_qubit_rotation(q, params[n_layers, i]) for i, q in enumerate(qubits))

    return circuit, list(params.flat), list(inputs.flat)


def entangling_layer(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops


def one_qubit_rotation(qubit, symbols):
    """
    Returns Cirq gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [cirq.rx(symbols[0])(qubit),
            cirq.ry(symbols[1])(qubit),
            cirq.rz(symbols[2])(qubit)]


class ReUploadingPQC(tf.keras.layers.Layer):
    """
    Performs the transformation (s_1, ..., s_d) -> (theta_1, ..., theta_N, lmbd[1][1]s_1, ..., lmbd[1][M]s_1,
        ......., lmbd[d][1]s_d, ..., lmbd[d][M]s_d) for d=input_dim, N=theta_dim and M=n_layers.
    An activation function from tf.keras.activations, specified by `activation` ('linear' by default) is
        then applied to all lmbd[i][j]s_i.
    All angles are finally permuted to follow the alphabetical order of their symbol names, as processed
        by the ControlledPQC.
    """

    def __init__(self, qubits, n_layers, observables, activation="linear", name="re-uploading_PQC"):
        super(ReUploadingPQC, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)
        circuit, theta_symbols, input_symbols = generate_circuit(qubits, n_layers)

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )

        lmbd_init = tf.ones(shape=(self.n_qubits * self.n_layers,))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas"
        )

        # Define explicit symbol order.
        symbols = [str(symb) for symb in theta_symbols + input_symbols]

        self.indices = tf.constant([sorted(symbols).index(a) for a in symbols])
        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])

        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)

    def call(self, inputs):
        # inputs[0] = encoding data for the state.
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)
        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_vars])


class Rescaling(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(Rescaling, self).__init__()
        self.input_dim = input_dim
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(1, input_dim)), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.math.multiply((inputs + 1) / 2, tf.repeat(self.w, repeats=tf.shape(inputs)[0], axis=0))


def generate_model_Qlearning(qubits, n_layers, n_actions, observables, target):
    """Generates a Keras model for a data re-uploading PQC Q-function approximator."""

    input_tensor = tf.keras.Input(shape=(len(qubits),), dtype=tf.dtypes.float32, name='input')
    re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables, activation='tanh')([input_tensor])
    process = tf.keras.Sequential([Rescaling(len(observables))], name=target * "Target" + "Q-values")
    Q_values = process(re_uploading_pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=Q_values)

    return model


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


def test_phase(env, model, epsilon, n_actions, steps_target_per_episode, repetitions):
    actions = [97, None, 100]
    wandb.define_metric("repetition")
    wandb.define_metric("test/*", step_metric="repetition")

    for r in range(repetitions):
        env.reset_game()

        for step in range(steps_target_per_episode + 1):

            state = np.array(list(env.getGameState()[0]))

            state_array = np.array(state)

            state = tf.convert_to_tensor([state_array])

            # Sample action
            coin = np.random.random()
            if coin > epsilon:
                q_vals = model([state])
                action = int(tf.argmax(q_vals[0]).numpy())
            else:
                action = np.random.choice(n_actions)

            action_names = ["left", "stay", "right"]
            reward = env.act(actions[action])

            if env.game_over():
                break
        wandb.log({'test/episode_reward': env.score(), "repetition": r})
        print(env.score())


def show_epopch(env, model, epsilon, n_actions, steps_target_per_episode, episode, save_dir, test_table, save=False, upload=False):
    actions = [97, None, 100]

    env.reset_game()
    print("\n")
    img = Image.new("RGB", (512, 512), (120, 120, 120))
    frames = []
    frames.append(img)
    frames.append(img)
    frames.append(img)

    for step in range(steps_target_per_episode + 1):

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

        action_names = ["left", "stay", "right"]
        reward = env.act(actions[action])

        # print(f"player x position {state_array[0]:3d},   fruits x position {state_array[1]:3d},   fruits y position {state_array[2]:3d}, action {action_names[action]}, reward {reward}, score {env.score()} ")
        print(f"player x position {int(state_array[0])},   fruits x position {int(state_array[1])},   fruits y position {state_array[2]:.2f}, action {action_names[action]}, reward {reward}, score {env.score()} ")

        if env.game_over():
            break
    print("Score: ", env.score(), "\n")
    if save:
        frames[1].save(f"{save_dir}/gifs/Simplified_catcher_epsiode_{episode}_steps_{step}.gif", save_all=True, append_images=frames[2:], optimize=False, loop=0)
    if upload:
        # ["Episode", "Video", "steps", "score"]
        test_table.add_data(episode, wandb.Video(f"{save_dir}/gifs/Simplified_catcher_epsiode_{episode}_steps_{step}.gif"), step, env.score())

        # wandb.log({"video": wandb.Video(f"{save_dir}/gifs/Simplified_catcher_epsiode_{episode}_steps_{step}.gif"), "episode":episode})


def get_parsed():
    parser = argparse.ArgumentParser(description="Define and run the experiment with one config")
    parser.add_argument('--run', type=int, help="Run iteration")
    parser.add_argument('--Quantum', action="store_true", default=False, help="True: Quantum model, False (default) classic model")

    arguments = vars(parser.parse_args())

    return arguments


def main():
    sleep(randint(0,5))
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    arguments = get_parsed()
    Quantum = arguments["Quantum"]

    name = f"Run-{arguments['run']}"
    project = "Catcher-Simplified"
    if Quantum:
        arg_mod = "Quantum_v8"
        type = "quantum"
        global tfq
        tfq = __import__('tensorflow_quantum', globals(), locals())
    else:
        arg_mod = "Classic_v2"
        type = "classic"

    message_name = f"{project}/{arg_mod}/{name}"
    save_dir = os.path.join("Saves", message_name)

    try:
        os.makedirs(save_dir)
    except OSError:
        print(" %s already exists or error" % save_dir)
    else:
        print("Successfully created the directory %s " % save_dir)

    try:
        os.makedirs(save_dir + "/gifs")
    except OSError:
        print(" %s already exists or error" % (save_dir + "/gifs"))
    else:
        print("Successfully created the directory %s " % (save_dir + "/gifs"))

    n_state = 3  # Dimension of the state vectors in CartPole
    n_actions = 3  # Number of actions in CartPole

    if Quantum:
        n_qubits = 3  # Dimension of the state vectors in CartPole     [player x position,   fruits x position,   fruits y position]
        n_layers = 15  # Number of layers in the PQC
        qubits = cirq.GridQubit.rect(1, n_qubits)
        ops = [cirq.Z(q) for q in qubits]
        observables = [ops[0], ops[1], ops[2]]  # Z_0*Z_1 for action 0 and Z_2*Z_3 for action 1
        model = generate_model_Qlearning(qubits, n_layers, n_actions, observables, False)
        model_target = generate_model_Qlearning(qubits, n_layers, n_actions, observables, True)
        batch_size = 16
        steps_per_update = 10  # Train the model every x steps
        steps_per_target_update = 30  # Update the target model every x steps
        n_episodes = 1000


    else:
        layers = [32, 32]  # layers = [9, 4] layers = [64] layers = [13] layers = [32, 32]
        model = generate_model(layers, n_state, n_actions)
        model_target = generate_model(layers, n_state, n_actions)
        batch_size = 32
        steps_per_update = 25  # Train the model every x steps
        steps_per_target_update = 75  # Update the target model every x steps
        n_episodes = 10000  # 30000

    model_target.set_weights(model.get_weights())
    print(model_target.summary())
    gamma = 0.99
    steps_target_per_episode = 250
    # Define replay memory
    max_memory_length = 10000  # Maximum replay length
    replay_memory = deque(maxlen=max_memory_length)

    epsilon = 1.0  # Epsilon greedy parameter
    epsilon_min = 0.01  # Minimum epsilon greedy parameter
    decay_epsilon = 0.95  # Decay rate of epsilon greedy parameter

    test_steps_target_per_episode = 1000
    test_repetitions = 100
    grad_log_steps = 100

    if Quantum:
        arguments = {
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "observables": observables,
            "gamma": gamma,
            "n_episodes": n_episodes,
            "steps_target_per_episode": steps_target_per_episode,
            "max_memory_length": max_memory_length,
            "epsilon": epsilon,
            "epsilon_min": epsilon_min,
            "decay_epsilon": decay_epsilon,
            "batch_size": batch_size,
            "steps_per_update": steps_per_update,
            "steps_per_target_update": steps_per_target_update,
            "test_steps_target_per_episode": test_steps_target_per_episode,
            "test_repetitions": test_repetitions,
            "grad_log_steps": grad_log_steps,
            "type": type,
            "run":arguments['run']
        }
    else:
        arguments = {
            "layers": layers,
            "gamma": gamma,
            "n_episodes": n_episodes,
            "steps_target_per_episode": steps_target_per_episode,
            "max_memory_length": max_memory_length,
            "epsilon": epsilon,
            "epsilon_min": epsilon_min,
            "decay_epsilon": decay_epsilon,
            "batch_size": batch_size,
            "steps_per_update": steps_per_update,
            "steps_per_target_update": steps_per_target_update,
            "test_steps_target_per_episode": test_steps_target_per_episode,
            "test_repetitions": test_repetitions,
            "grad_log_steps": grad_log_steps,
            "type": type,
            "run": arguments['run']

        }

    run = wandb.init(project=project,
                     group=arg_mod,
                     name=arg_mod + "_" + name,
                     config=arguments,
                     dir=save_dir)

    # create a Table with the same columns as above,
    # plus confidence scores for all labels
    columns = ["Episode", "Video", "steps", "score"]

    test_table = wandb.Table(columns=columns)

    # run inference on every image, assuming my_model returns the
    # predicted label, and the ground truth labels are available

    wandb.alert(title="Experiment Started", text=f"Experiment {message_name} Started", wait_duration=timedelta(seconds=0))

    wandb.define_metric("training_update/step")
    grad_names = [f"training_update/{n.name}" for n in model.trainable_variables] + ["training_update/step", "training_update/episode"]
    wandb.define_metric("training_update/*", step_metric="repetition")

    wandb.define_metric("episode")
    wandb.define_metric("episode_reward", step_metric="episode")
    if Quantum:
        optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
        optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
        optimizer_out = tf.keras.optimizers.Adam()#learning_rate=0.01, amsgrad=True)
        # Assign the model parameters to each optimizer
        w_in, w_var, w_out = 1, 0, 2
    else:
        optimizer = tf.keras.optimizers.Adam()  # learning_rate=1e-2)

    game = Catcher(width=512, height=512, init_lives=3)
    p = PLE(game, display_screen=False, state_preprocessor=process_state)  # ,  force_fps=False, fps=240)
    env = PLE(game, display_screen=True, state_preprocessor=process_state)  # , force_fps=False, fps=2)

    episode_reward_history = []
    episode_score_history = []

    steps_per_ep_history = []

    step_count = 0
    t = trange(n_episodes, desc='Training', leave=True)
    train_updates = 0
    best_avg_100_score = 0

    for episode in t:
        episode_reward = 0
        p.reset_game()
        state = np.array(list(p.getGameState()[0]))
        steps_in_episode = 0

        for step in range(steps_target_per_episode + 1):
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
                train_updates += 1
                # Sample a batch of interactions and update Q_function
                training_batch = np.random.choice(replay_memory, size=batch_size)
                if Quantum:
                    grads = Q_learning_update_quantum(np.asarray([x['state'] for x in training_batch]),
                                                      np.asarray([x['action'] for x in training_batch]),
                                                      np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
                                                      np.asarray([x['next_state'] for x in training_batch]),
                                                      np.asarray([x['done'] for x in training_batch], dtype=np.float32),
                                                      model, model_target, gamma, n_actions, optimizer_in, optimizer_var, optimizer_out, w_in, w_var, w_out)

                else:
                    grads = Q_learning_update_classic(np.asarray([x['state'] for x in training_batch]),
                                                      np.asarray([x['action'] for x in training_batch]),
                                                      np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
                                                      np.asarray([x['next_state'] for x in training_batch]),
                                                      np.asarray([x['done'] for x in training_batch], dtype=np.float32),
                                                      model, model_target, gamma, optimizer, n_actions)

                grad_list = [wandb.Histogram(g) for g in grads] + [train_updates, episode]
                if train_updates % 100 == 0:
                    wandb.log(dict(zip(grad_names, grad_list)))

            # Update target model
            if step_count % steps_per_target_update == 0:
                model_target.set_weights(model.get_weights())

            # Check if the episode is finished
            if interaction['done']:
                break

        if episode % (n_episodes // 10) == 0:
            show_epopch(env, model_target, epsilon, n_actions, steps_target_per_episode, episode, save_dir, test_table, True, True)

        # Decay epsilon
        epsilon = max(epsilon * decay_epsilon, epsilon_min)
        episode_reward_history.append(episode_reward)
        wandb.log({'episode_reward': episode_reward, "steps_in_episode": steps_in_episode, "episode": episode})
        avg_rewards = np.mean(episode_reward_history[-10:])

        avg_rewards_100 = np.mean(episode_reward_history[-100:])
        # Save model if avg score of las 100 is the best
        if avg_rewards_100 >= best_avg_100_score:
            wandb.run.summary["Saved_at_episode"] = episode
            wandb.run.summary["Saved_avg_score"] = avg_rewards_100
            wandb.run.summary["Saved_avg"] = p.score()
            model_target.save_weights(save_dir + "/saved_weights/target_weights")
            best_avg_100_score = avg_rewards_100

        t.set_description("Episode {:5d}/{:5d}, steps in ep {:4d}, score {:04.2f}, avg 10 rew {:04.2f}, avg 100 rew {:04.2f}".format(episode + 1, n_episodes, steps_in_episode, p.score(), avg_rewards, avg_rewards_100))
        t.refresh()  # to show immediately the update
        # if avg_rewards_200 > (steps_target_per_episode//5 )*.95:
        #     break

    # plt.plot(episode_score_history)
    # plt.title("Score per trial")
    # plt.show()
    # plt.plot(episode_reward_history)
    # plt.title("Reward per trial")
    # plt.show()
    # plt.title("Steps per trial")
    # plt.plot(steps_per_ep_history)
    # plt.show()
    # Testing
    wandb.run.summary["Final_total_interaction_steps"] = step_count
    wandb.run.summary["Final_train_updates"] = train_updates

    # For visualisation reset episode
    final_comp_table = wandb.Table(columns=columns)
    episode = 1

    show_epopch(env, model_target, epsilon, n_actions, steps_target_per_episode, episode, save_dir, final_comp_table, True, True)

    model_target.load_weights(save_dir + "/saved_weights/target_weights")

    show_epopch(env, model_target, epsilon, n_actions, steps_target_per_episode, episode + 1, save_dir, final_comp_table, True, True)
    show_epopch(env, model_target, epsilon, n_actions, test_steps_target_per_episode, episode + 2, save_dir, final_comp_table, True, True)

    wandb.log({"video_table": test_table, "final_video_table": final_comp_table})
    test_phase(p, model_target, epsilon, n_actions, test_steps_target_per_episode, test_repetitions)


    wandb.save(f"Saves/logs_to_move/run-{arguments['run']}.log")

    wandb.alert(title="Experiment finished", text=f"Experiment {message_name} ended", wait_duration=timedelta(seconds=0))

    run.finish()


if __name__ == "__main__":
    main()
