import argparse
import os
import sys
from collections import deque
from datetime import timedelta
from random import randint
from time import sleep
import time

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
    return grads, loss


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
        # loss = tf.keras.losses.mean_squared_error(target_q_values, q_values_masked)

        # Backpropagation
    grads = tape.gradient(loss, model.trainable_variables)

    for optimizer, w in zip([optimizer_in, optimizer_var, optimizer_out], [w_in, w_var, w_out]):
        optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])
    return grads, loss


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


def test_phase(env, model, epsilon, n_actions, steps_target_per_episode, repetitions, arguments):
    actions = [97, None, 100]

    wandb.define_metric("repetition")
    wandb.define_metric("test/*", step_metric="repetition")
    my_table = wandb.Table(columns=["repetition", "reward", "name", "group"])

    for r in trange(repetitions, desc='Test phase'):
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
        my_table.add_data(r, env.score(), arguments["name"], arguments["group"])

    wandb.log({"Test predictions": my_table})


def show_epopch(env, model, epsilon, n_actions, steps_target_per_episode, episode, save_dir, test_table, save=False, upload=False):
    actions = [97, None, 100]

    env.reset_game()
    print("\n")
    img = Image.new("RGB", (512, 512), (120, 120, 120))
    frames = [img, img, img]

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
    parser.add_argument('--run', '-r', type=int, help="Run iteration")
    parser.add_argument('--Quantum', '-Q', action="store_true", default=False, help="True: Quantum model, False (default) classic model")
    parser.add_argument('--n_layers', '-Ql', type=int, help=" number of layers for the Quantum model")
    parser.add_argument('--layers', '-l', nargs='+', type=int, help=" number of layers for the Classic model. for example 32 16 means 2 layers with first one with 32 neurons second with 16 neurons")
    parser.add_argument('--name', '-n', type=str, help="name of the experiment")

    arguments = vars(parser.parse_args())

    return arguments


def main():
    arguments = get_parsed()

    Quantum = arguments["Quantum"]
    print(arguments)
    if Quantum:
        sleep(30)

    name = f"Run-{arguments['run']}"
    project = "Catcher-Simplified"
    if Quantum:
        tf.config.set_visible_devices([], 'GPU')
        arg_mod = arguments["name"]  # Quantum_v10_cpu"  # "Quantum_v9_cpu"
        type = "quantum"
        global tfq
        tfq = __import__('tensorflow_quantum', globals(), locals())
    else:
        arg_mod = arguments["name"]  # "Classic_v3"
        type = "classic"
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
        n_layers = arguments["n_layers"]  # 10 #15 # Number of layers in the PQC
        qubits = cirq.GridQubit.rect(1, n_qubits)
        ops = [cirq.Z(q) for q in qubits]
        observables = [ops[0], ops[1], ops[2]]  # Z_0*Z_1 for action 0 and Z_2*Z_3 for action 1
        model = generate_model_Qlearning(qubits, n_layers, n_actions, observables, False)
        model_target = generate_model_Qlearning(qubits, n_layers, n_actions, observables, True)
        batch_size = 16
        steps_per_update = 10  # Train the model every x steps
        steps_per_target_update = 30  # Update the target model every x steps

    else:
        layers = arguments["layers"]  # [32, 32]  # layers = [9, 4] layers = [64] layers = [13] layers = [32, 32]
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
    training_stop_episodes = 100
    training_stop_percent = 0.98

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
            "training_stop_episodes": training_stop_episodes,
            "training_stop_percent": training_stop_percent,
            "run": arguments['run'],
            "name": arg_mod + "_" + name,
            "group": arg_mod
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
            "training_stop_episodes": training_stop_episodes,
            "training_stop_percent": training_stop_percent,
            "run": arguments['run'],
            "name": arg_mod + "_" + name,
            "group": arg_mod
        }

    run = wandb.init(project=project,
                     group=arg_mod,
                     name=arg_mod + "_" + name,
                     config=arguments,
                     dir="Saves")

    # create a Table with the same columns as above,
    # plus confidence scores for all labels
    columns = ["Episode", "Video", "steps", "score"]

    # test_table = wandb.Table(columns=columns)

    # run inference on every image, assuming my_model returns the
    # predicted label, and the ground truth labels are available

    # wandb.alert(title="Experiment Started", text=f"Experiment {message_name} Started", wait_duration=timedelta(seconds=0))

    wandb.define_metric("training_metrics/episode")
    wandb.define_metric("training_metrics/*", step_metric="training_metrics/episode")

    if Quantum:
        optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
        optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
        optimizer_out = tf.keras.optimizers.Adam()  # learning_rate=0.01, amsgrad=True)
        # Assign the model parameters to each optimizer
        w_in, w_var, w_out = 1, 0, 2

        wandb.define_metric("training_update_quantum/step")
        wandb.define_metric("training_update_quantum/*", step_metric="training_update_quantum/step")
        grad_names = [f"training_update_quantum/{n.name}_grads" for n in model.trainable_variables] + ["training_update_quantum/step", "training_update_quantum/episode", "training_update_quantum/loss"]

        wandb.define_metric("parameters/*", step_metric="training_update_quantum/step")
        # param_names = [f"parameters/{n.name}/{n.name}_{i}" for n in model_target.trainable_variables for i in range(n.shape[-1])]  + ["parameters/step" , "parameters/episode]
        # param_names = [f"parameters/unfiltered/{n.name}/{n.name}_{i}" for n in model_target.trainable_variables for i in range(n.shape[-1])] + [f"parameters/filtered/{model_target.trainable_variables[0].name}/{model_target.trainable_variables[0].name}_{i}" for i in range(model_target.trainable_variables[0].shape[-1])] + ["parameters/step","parameters/episode"]
        # param_names = [f"parameters/{n.name}/{n.name}_{i}" for n in model_target.trainable_variables for i in range(n.shape[-1])] + ["training_update_quantum/step", "training_update_quantum/episode"]
        param_names = [f"parameters/{n.name}/{n.name}_{i}" for n in model_target.trainable_variables[1:] for i in range(n.shape[-1])] + ["training_update_quantum/step", "training_update_quantum/episode"]

        [print(f"{n.name[:-2]} shape = {n.shape}") for n in model_target.trainable_variables]
    else:
        optimizer = tf.keras.optimizers.Adam()  # learning_rate=1e-2)

        wandb.define_metric("training_update_classic/step")
        wandb.define_metric("training_update_classic/*", step_metric="training_update_classic/step")
        grad_names = [f"training_update_classic/{n.name}_grads" for n in model.trainable_variables] + ["training_update_classic/step", "training_update_classic/episode", "training_update_classic/loss"]

    game = Catcher(width=512, height=512, init_lives=3)
    p = PLE(game, display_screen=False, state_preprocessor=process_state)  # ,  force_fps=False, fps=240)
    env = PLE(game, display_screen=True, state_preprocessor=process_state)  # , force_fps=False, fps=2)

    episode_reward_history = []

    steps_per_ep_history = []

    step_count = 0
    t = trange(n_episodes, desc='Training', leave=True)
    train_updates = 0
    best_avg_100_score = 0
    training_stop_counter = 0
    plot_vals_x = []
    stopped_at = n_episodes

    for episode in t:
        start_ep_time = time.time()
        if training_stop_counter < training_stop_episodes:
            episode_reward = 0
            p.reset_game()
            state = np.array(list(p.getGameState()[0]))
            steps_in_episode = 0

            for step in range(steps_target_per_episode + 1):
                start = time.time()
                # Interact with env
                interaction = interact_env(state, model, epsilon, n_actions, p)

                # Store interaction in the replay memory
                replay_memory.append(interaction)

                state = interaction['next_state']
                episode_reward += interaction['reward']
                step_count += 1
                steps_in_episode += 1
                interaction_time = (time.time() - start)
                # Update model

                if step_count % steps_per_update == 0:
                    update_start = time.time()
                    train_updates += 1
                    # Sample a batch of interactions and update Q_function
                    training_batch = np.random.choice(replay_memory, size=batch_size)
                    if Quantum:

                        grads, loss = Q_learning_update_quantum(np.asarray([x['state'] for x in training_batch]),
                                                                np.asarray([x['action'] for x in training_batch]),
                                                                np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
                                                                np.asarray([x['next_state'] for x in training_batch]),
                                                                np.asarray([x['done'] for x in training_batch], dtype=np.float32),
                                                                model, model_target, gamma, n_actions, optimizer_in, optimizer_var, optimizer_out, w_in, w_var, w_out)
                        update_time = (time.time() - update_start)
                        start_log_time = time.time()
                        grad_list = [np.sum(np.abs(g)) for g in grads] + [train_updates, episode, loss]  # mean or sum
                        if len(episode_reward_history) > 0:
                            grad_names += ["training_update_quantum/last_ep_reward"]
                            grad_list = grad_list + [episode_reward_history[-1]]
                        wandb.log(dict(zip(grad_names, grad_list)))

                        if step_count % (steps_per_target_update * 10) == 0:
                            # params_target_unf = model_target.trainable_variables[0][0].numpy().tolist() + model_target.trainable_variables[1].numpy().tolist() + model_target.trainable_variables[2][0].numpy().tolist()  # thetas, lambdas, obs-weights
                            # params_target_f = np.mod(model_target.trainable_variables[0][0].numpy()+0.0000001, 2 * np.pi).tolist()
                            # params = params_target_unf + params_target_f + [train_updates, episode]
                            # params = np.mod(np.rad2deg(model_target.trainable_variables[0][0].numpy()), 360).tolist() + model_target.trainable_variables[1].numpy().tolist() + model_target.trainable_variables[2][0].numpy().tolist() + [train_updates, episode]  # thetas, lambdas, obs-weights
                            params = model_target.trainable_variables[1].numpy().tolist() + model_target.trainable_variables[2][0].numpy().tolist() + [train_updates, episode]  # thetas, lambdas, obs-weights

                            if len(episode_reward_history) > 0:
                                param_names += ["parameters/last_ep_reward"]
                                params = params + [episode_reward_history[-1]]
                            wandb.log(dict(zip(param_names, params)))
                            plot_vals_x.append(train_updates)
                            if len(plot_vals_x) == 1:
                                plot_vals_y = np.mod(np.rad2deg(model_target.trainable_variables[0][0].numpy()), 360)
                            else:
                                plot_vals_y = np.vstack((plot_vals_y, np.mod(np.rad2deg(model_target.trainable_variables[0][0].numpy()), 360)))


                    else:
                        grads, loss = Q_learning_update_classic(np.asarray([x['state'] for x in training_batch]),
                                                                np.asarray([x['action'] for x in training_batch]),
                                                                np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
                                                                np.asarray([x['next_state'] for x in training_batch]),
                                                                np.asarray([x['done'] for x in training_batch], dtype=np.float32),
                                                                model, model_target, gamma, optimizer, n_actions)
                        update_time = (time.time() - update_start)
                        start_log_time = time.time()
                        grad_list = [np.sum(np.abs(g)) for g in grads] + [train_updates, episode, loss]  # mean or sum
                        if len(episode_reward_history) > 0:
                            grad_names += ["training_update_classic/last_ep_reward"]
                            grad_list = grad_list + [episode_reward_history[-1]]
                        wandb.log(dict(zip(grad_names, grad_list)))

                    wandb.log({  # only logged for training updates otherwise logging breaks. To much data.
                        "time/step": step_count,
                        "time/update_time": update_time,
                        "time/logging_time": time.time() - start_log_time,
                        "time/interaction_time": interaction_time
                    })

                # Update target model
                if step_count % steps_per_target_update == 0:
                    model_target.set_weights(model.get_weights())

                # Check if the episode is finished
                if interaction['done']:
                    break

            if episode % (n_episodes // 10) == 0:
                show_epopch(env, model_target, epsilon, n_actions, steps_target_per_episode, episode, save_dir, None, True, False)

            start_ep_log_time = time.time()
            # Decay epsilon
            epsilon = max(epsilon * decay_epsilon, epsilon_min)
            episode_reward_history.append(episode_reward)
            avg_rewards_100 = np.mean(episode_reward_history[-100:])
            # Save model if avg score of las 100 is the best
            if avg_rewards_100 >= best_avg_100_score:
                wandb.run.summary["Saved_at_episode"] = episode
                wandb.run.summary["Saved_avg_score"] = avg_rewards_100
                wandb.run.summary["Saved_avg"] = p.score()
                model_target.save_weights(save_dir + "/saved_weights/target_weights")
                best_avg_100_score = avg_rewards_100
            if avg_rewards_100 >= (steps_target_per_episode // 5) * training_stop_percent:
                training_stop_counter += 1
            else:
                training_stop_counter = 0

            stopped_at = episode

        else:
            start_ep_log_time = time.time()

            episode_reward = avg_rewards_100

            episode_reward_history.append(episode_reward)

        avg_rewards = np.mean(episode_reward_history[-10:])

        avg_rewards_100 = np.mean(episode_reward_history[-100:])

        wandb.log({'training_metrics/episode_reward': episode_reward,
                   'training_metrics/episode_reward_avg_10': avg_rewards,
                   'training_metrics/episode_reward_avg_100': avg_rewards_100,
                   "training_metrics/steps_in_episode": steps_in_episode,
                   "training_metrics/training_stop_counter": training_stop_counter,
                   "training_metrics/epsilon": epsilon,
                   "training_metrics/episode": episode,
                   "time/episode_time": time.time() - start_ep_time,
                   "time/ep_log_time": time.time() - start_ep_log_time,
                   })

        if training_stop_counter < training_stop_episodes:
            t.set_description("Episode {:5d}/{:5d}, steps in ep {:4d}, score {:04.2f}, avg 10 rew {:04.2f}, avg 100 rew {:04.2f}, training_stop_counter {:3d}".format(episode + 1, n_episodes, steps_in_episode, episode_reward, avg_rewards, avg_rewards_100, training_stop_counter))
        else:
            t.set_description("Episode {:5d}/{:5d}, steps in ep {:4d}, score {:04.2f}, avg 10 rew {:04.2f}, avg 100 rew {:04.2f}, stopped at {:3d}".format(episode + 1, n_episodes, steps_in_episode, episode_reward, avg_rewards, avg_rewards_100, stopped_at))

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

    # For visualisation reset episode
    final_comp_table = wandb.Table(columns=columns)
    episode = 1

    show_epopch(env, model_target, epsilon, n_actions, steps_target_per_episode, episode, save_dir, final_comp_table, True, True)

    model_target.load_weights(save_dir + "/saved_weights/target_weights")

    if Quantum:
        theta_names = [f"{model_target.trainable_variables[0].name}_{i}" for i in range(model_target.trainable_variables[0].shape[-1])]
        thetas_table = wandb.Table(columns=["theta_val", "theta_name", "name"])
        for t, n in zip(np.mod(np.rad2deg(model_target.trainable_variables[0][0].numpy()), 360), theta_names):
            thetas_table.add_data(t, n, arguments["name"])

        lambdas_table = wandb.Table(columns=["lambda_val", "lambda_name", "name"])
        for t, n in zip(np.mod(np.rad2deg(model_target.trainable_variables[1].numpy()), 360), [f"{model_target.trainable_variables[1].name}_{i}" for i in range(model_target.trainable_variables[1].shape[-1])]):
            lambdas_table.add_data(t, n, arguments["name"])

        obs_weights_table = wandb.Table(columns=["obs_weights_val", "obs_weights_name", "name"])
        for t, n in zip(np.mod(np.rad2deg(model_target.trainable_variables[2][0].numpy()), 360), [f"{model_target.trainable_variables[2].name}_{i}" for i in range(model_target.trainable_variables[2].shape[-1])]):
            obs_weights_table.add_data(t, n, arguments["name"])

        wandb.log({"thetas_final": thetas_table,
                   "lambdas_final": lambdas_table,
                   "obs_weights_table": obs_weights_table,
                   "thetas_table": wandb.plot.line_series(xs=plot_vals_x, ys=[plot_vals_y[:, i].tolist() for i in range(plot_vals_y.shape[1])], keys=theta_names, title="Thetas values over time", xname="Training updates")
                   })

    show_epopch(env, model_target, epsilon, n_actions, steps_target_per_episode, episode + 1, save_dir, final_comp_table, True, True)
    show_epopch(env, model_target, epsilon, n_actions, test_steps_target_per_episode, episode + 2, save_dir, final_comp_table, True, True)

    # wandb.log({"video_table": test_table, "final_video_table": final_comp_table})
    wandb.log({"final/video_table": final_comp_table,
               "final/total_interaction_steps": step_count,
               "final/train_updates": train_updates,
               "final/stopped_at": stopped_at,
               })

    test_phase(p, model_target, epsilon, n_actions, test_steps_target_per_episode, test_repetitions, arguments)
    wandb.save(f"{save_dir}/err.log")
    wandb.save(f"{save_dir}/out.log")

    # wandb.alert(title="Experiment finished", text=f"Experiment {message_name} ended", wait_duration=timedelta(seconds=0))

    run.finish()


if __name__ == "__main__":
    main()
