import argparse
import sys
from collections import deque

import cirq
import gym
import numpy as np
import sympy
import tensorflow as tf

from tqdm import trange

import wandb
import os


def get_parsed(text=None):
    parser = argparse.ArgumentParser(description="Define and run the experiment with one config")
    parser.add_argument('--run', type=int, help="")
    parser.add_argument('--repetitions', type=int, default=None, help="")
    parser.add_argument('--loss', type=str, default="mse")
    parser.add_argument('--observables', type=int, help="")
    parser.add_argument('--source', type=str, default="TFQ", help="From which textual source the circuit was copied")

    if text is not None:
        arguments = vars(parser.parse_args(text))
    else:
        arguments = vars(parser.parse_args())

    return arguments


def one_qubit_rotation(qubit, symbols):
    """
    Returns Cirq gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [cirq.rx(symbols[0])(qubit),
            cirq.ry(symbols[1])(qubit),
            cirq.rz(symbols[2])(qubit)]


def one_qubit_rotation_paper(qubit, symbols):
    """
    Returns Cirq gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [cirq.ry(symbols[0])(qubit),
            cirq.rz(symbols[1])(qubit)]


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


def generate_circuit_paper(qubits, n_layers):
    """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
    # Number of qubits
    n_qubits = len(qubits)

    # Sympy symbols for variational angles
    params = sympy.symbols(f'theta(0:{2 * (n_layers) * n_qubits})')
    params = np.asarray(params).reshape((n_layers, n_qubits, 2))

    # Sympy symbols for encoding angles
    inputs = sympy.symbols(f'x(0:{n_qubits})' + f'(0:{n_layers})')
    inputs = np.asarray(inputs).reshape((n_layers, n_qubits))

    # Sympy symbols for post rotation angles
    params_final = sympy.symbols(f'theta_end(0:{2 * n_qubits})')
    params_final = np.asarray(params_final).reshape((1, n_qubits, 2))

    # Define circuit
    circuit = cirq.Circuit()
    for l in range(n_layers):
        # Encoding layer
        circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(qubits))
        # Variational layer
        circuit += cirq.Circuit(one_qubit_rotation_paper(q, params[l, i]) for i, q in enumerate(qubits))
        circuit += entangling_layer(qubits)

    # Last varitional layer
    circuit += cirq.Circuit(one_qubit_rotation_paper(q, params_final[0, i]) for i, q in enumerate(qubits))

    return circuit, list(params.flat), list(inputs.flat), list(params_final.flat)


def entangling_layer(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops


class ReUploadingPQC(tf.keras.layers.Layer):
    """
    Performs the transformation (s_1, ..., s_d) -> (theta_1, ..., theta_N, lmbd[1][1]s_1, ..., lmbd[1][M]s_1,
        ......., lmbd[d][1]s_d, ..., lmbd[d][M]s_d) for d=input_dim, N=theta_dim and M=n_layers.
    An activation function from tf.keras.activations, specified by `activation` ('linear' by default) is
        then applied to all lmbd[i][j]s_i.
    All angles are finally permuted to follow the alphabetical order of their symbol names, as processed
        by the ControlledPQC.
    """

    def __init__(self, qubits, n_layers, observables, repetitions, source="TFQ", activation="linear", name="re-uploading_PQC"):
        super(ReUploadingPQC, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)
        if source == "Paper":
            circuit, theta_symbols, input_symbols, final_vals = generate_circuit_paper(qubits, n_layers)
            init_val = tf.constant([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2], dtype="float32", shape=(1, self.n_qubits * 2))
            self.final_vals_var = tf.Variable(initial_value=init_val, dtype="float32", shape=(1, len(final_vals)), trainable=False, name="Post rotation")
            self.source = "Paper"
        else:
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
        if source == "Paper":
            symbols = [str(symb) for symb in theta_symbols + input_symbols + final_vals]
        else:
            symbols = [str(symb) for symb in theta_symbols + input_symbols]

        self.indices = tf.constant([sorted(symbols).index(a) for a in symbols])
        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])

        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables, repetitions=repetitions)


    def call(self, inputs):
        # inputs[0] = encoding data for the state.
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        if self.source == "Paper":
            tiled_up_final_vals_var = tf.tile(self.final_vals_var, multiples=[batch_dim, 1])
            joined_vars = tf.concat([tiled_up_thetas, squashed_inputs, tiled_up_final_vals_var], axis=1)
        else:
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


def generate_model_Qlearning(qubits, n_layers, n_actions, observables, target, reps, source):
    """Generates a Keras model for a data re-uploading PQC Q-function approximator."""

    input_tensor = tf.keras.Input(shape=(len(qubits),), dtype=tf.dtypes.float32, name='input')
    re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables, source=source, repetitions=reps, activation='tanh')([input_tensor])
    process = tf.keras.Sequential([Rescaling(len(observables))], name=target * "Target" + "Q-values")
    Q_values = process(re_uploading_pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=Q_values)

    return model


def interact_env(state, model, epsilon, n_actions, env):
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

    # Apply sampled action in the environment, receive reward and next state
    next_state, reward, done, _ = env.step(action)

    interaction = {'state': state_array, 'action': action, 'next_state': next_state.copy(),
                   'reward': reward, 'done': float(done)}

    return interaction


# @tf.function
def Q_learning_update(states, actions, rewards, next_states, done, model, model_target, gamma, n_actions, optimizer_in, optimizer_var, optimizer_out, w_in, w_var, w_out, loss_name):
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
        if loss_name == 'mse':
            loss = tf.keras.losses.mean_squared_error(target_q_values, q_values_masked)
        else:
            loss = tf.keras.losses.Huber()(target_q_values, q_values_masked)  # Original

    # Backpropagation
    grads = tape.gradient(loss, model.trainable_variables)
    for optimizer, w in zip([optimizer_in, optimizer_var, optimizer_out], [w_in, w_var, w_out]):
        optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])


def main():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    global tfq
    tfq = __import__('tensorflow_quantum', globals(), locals())

    arguments = get_parsed()

    project = "CartPole-V1"
    if arguments['repetitions'] is None:
        group = "QML-Analytical_v2"
    else:
        group = "QML-Simulated_v2"

    if arguments['source'] == "Paper":
        group = "Paper_" + group

    loss_name = arguments['loss']
    group = group + "_" + loss_name + "_obs_" + str(arguments["observables"])

    name = "run-" + str(arguments["run"])
    run = wandb.init(project=project,
                     group=group,
                     name=group + "_" + name,
                     config=arguments)
    save_dir = os.path.join(project, group, name)

    n_qubits = 4  # Dimension of the state vectors in CartPole
    n_layers = 5  # Number of layers in the PQC
    n_actions = 2  # Number of actions in CartPole

    qubits = cirq.GridQubit.rect(1, n_qubits)
    ops = [cirq.Z(q) for q in qubits]
    if arguments['observables'] == 2:
        observables = [ops[0] * ops[1], ops[2] * ops[3]]  # Z_0*Z_1 for action 0 and Z_2*Z_3 for action 1
    elif arguments['observables'] == 0:
        observables = [ops[0], ops[0]]
    elif arguments['observables'] == 3:
        observables = [ops[0], ops[1] * ops[2] * ops[3]]
    elif arguments['observables'] == 1:
        observables = [ops[0], ops[3]]
    elif arguments['observables'] == 4:
        observables = [ops[0], ops[2]]
    elif arguments['observables'] == 5:
        observables = [ops[0] * ops[2], ops[1] * ops[3]]
    else:
        sys.exit("No observables declared")

    model = generate_model_Qlearning(qubits, n_layers, n_actions, observables, False, arguments["repetitions"], arguments['source'])
    model_target = generate_model_Qlearning(qubits, n_layers, n_actions, observables, True, arguments["repetitions"], arguments['source'])
    model_target.set_weights(model.get_weights())
    print(model_target.summary())
    gamma = 0.99
    n_episodes = 2000

    # Define replay memory
    max_memory_length = 10000  # Maximum replay length
    replay_memory = deque(maxlen=max_memory_length)

    epsilon = 1.0  # Epsilon greedy parameter
    epsilon_min = 0.01  # Minimum epsilon greedy parameter
    decay_epsilon = 0.99  # Decay rate of epsilon greedy parameter
    batch_size = 16
    steps_per_update = 10  # Train the model every x steps
    steps_per_target_update = 30  # Update the target model every x steps

    optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
    optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
    optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)

    # Assign the model parameters to each optimizer
    w_in, w_var, w_out = 1, 0, 2

    env = gym.make("CartPole-v1")

    episode_reward_history = []
    step_count = 0
    avg_rewards_100 = 0
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
                                  model, model_target, gamma, n_actions, optimizer_in, optimizer_var, optimizer_out, w_in, w_var, w_out, loss_name)
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
