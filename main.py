from collections import deque
import random
import pygame

import numpy as np
import gym  # Contains the game we want to play
from tensorflow.keras.models import Sequential  # To compose multiple Layers
from tensorflow.keras.layers import Dense  # Fully-Connected layer
from tensorflow.keras.layers import Activation  # Activation functions
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model

EPOCHS = 1000
EPSILON_REDUCE = 0.995
LEARNING_RATE = 0.001  # Not the same as alpha in Q-learning
GAMMA = 0.95


def create_env():
    env_name = 'CartPole-v1'
    env = gym.make(env_name, render_mode='human')
    return env


def create_nn(env):
    num_of_actions = env.action_space.n  # 2 actions for CartPole - Move left or Move right
    num_observations = env.observation_space.shape[0]  # 4 observations for CartPole

    model = Sequential()
    model.add(Dense(16, input_shape=(1, num_observations)))  # 4 input neurons to 16 in a dense layer
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(num_of_actions))  # Final output layer of two actions
    model.add(Activation('linear'))

    print(model.summary())
    return model


def epsilon_greedy_action_selection(env, model, epsilon, observation):
    if np.random.random() > epsilon:
        prediction = model.predict(observation)  # perform the prediction on the observation
        action = np.argmax(prediction)  # Chose the action with the higher value
    else:
        action = np.random.randint(0, env.action_space.n)  # Else use random action
    return action


def replay(replay_buffer, batch_size, model, target_model):
    # As long as the buffer has not enough elements we do nothing
    if len(replay_buffer) < batch_size:
        return

    # Take a random sample from the buffer with size batch_size
    samples = random.sample(replay_buffer, batch_size)
    target_batch = []  # to store the targets predicted by the target network for training

    # Efficient way to handle the sample by using the zip functionality
    zipped_samples = list(zip(*samples))
    states, actions, rewards, new_states, dones = zipped_samples

    targets = target_model.predict(np.array(states))  # Predict targets for all states from the sample
    q_values = model.predict(np.array(new_states))  # Predict Q-Values for all new states from the sample

    # Now we loop over all predicted values to compute the actual targets
    for i in range(batch_size):
        q_value = max(q_values[i][0])  # Take the maximum Q-Value for each sample

        # Store the ith target in order to update it according to the formula
        target = targets[i].copy()
        if dones[i]:
            target[0][actions[i]] = rewards[i]
        else:
            target[0][actions[i]] = rewards[i] + q_value * GAMMA
        target_batch.append(target)

    # Fit the model based on the states and the updated targets for 1 epoch
    model.fit(np.array(states), np.array(target_batch), epochs=1, verbose=0)


def run_environment():
    epsilon = 1.0
    env = create_env()
    model = create_nn(env)
    target_model = clone_model(model)

    replay_buffer = deque(maxlen=20000)
    update_target_model = 10





run_environment()


# ### deque examples
# deque_1 = deque(maxlen=5)
# for i in range(5):  # all values fit into the deque, no overwriting
#     deque_1.append(i)
# print(deque_1)
# print("---------------------")
# deque_2 = deque(maxlen=5)
#
# # after the first 5 values are stored, it needs to overwrite the oldest value to store the new one
# for i in range(10):
#     deque_2.append(i)
#     print(deque_2)