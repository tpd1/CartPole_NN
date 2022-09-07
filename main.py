from collections import deque
import random
import pygame
import IPython

import numpy as np
import gym  # Contains the game we want to play
from tensorflow.keras.models import Sequential  # To compose multiple Layers
from tensorflow.keras.layers import Dense  # Fully-Connected layer
from tensorflow.keras.layers import Activation  # Activation functions
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model

EPOCHS = 200
EPSILON_REDUCE = 0.995
LEARNING_RATE = 0.001  # Not the same as alpha in Q-learning
GAMMA = 0.95


def create_env():
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    return env


def create_nn(env):
    num_of_actions = env.action_space.n  # 2 actions for CartPole - Move left or Move right
    num_observations = env.observation_space.shape[0]  # 4 observations for CartPole
    print(f"Num of observations:  {num_observations}, num of actions: {num_of_actions}")

    model = Sequential()
    model.add(Dense(16, input_shape=(1, num_observations)))  # 4 input neurons to 16 in a dense layer
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(num_of_actions))  # Final output layer of two actions
    model.add(Activation('linear'))

    return model


def epsilon_greedy_action_selection(env, model, epsilon, observation):
    if np.random.random() > epsilon:
        prediction = model.predict(observation.reshape((1,1,4)), verbose=0)  # perform the prediction on the observation
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

    targets = target_model.predict(np.array(states), verbose=0)  # Predict targets for all states from the sample
    q_values = model.predict(np.array(new_states), verbose=0)  # Predict Q-Values for all new states from the sample

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


def update_model_handler(epoch, update_target_model, model, target_model):
    if epoch > 0 and epoch % update_target_model == 0:
        target_model.set_weights(model.get_weights())


def run_environment():
    epsilon = 1.0
    env = create_env()
    model = create_nn(env)
    target_model = clone_model(model)

    replay_buffer = deque(maxlen=20000)
    update_target_model = 10

    model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))

    best_so_far = 0
    for epoch in range(EPOCHS):
        observation = env.reset()  # Get inital state

        # Keras expects the input to be of shape [1, X] thus we have to reshape
        observation = observation.reshape([1, 4])
        done = False

        points = 0
        while not done:  # as long current run is active

            # Select action acc. to strategy
            action = epsilon_greedy_action_selection(env, model, epsilon, observation)

            # Perform action and get next state
            next_observation, reward, done, info = env.step(action)
            next_observation = next_observation.reshape([1, 4])  # Reshape!!
            replay_buffer.append((observation, action, reward, next_observation, done))  # Update the replay buffer
            observation = next_observation  # update the observation
            points += 1

            # Most important step! Training the model by replaying
            replay(replay_buffer, 32, model, target_model)

        epsilon *= EPSILON_REDUCE  # Reduce epsilon

        # Check if we need to update the target model
        update_model_handler(epoch, update_target_model, model, target_model)

        if points > best_so_far:
            best_so_far = points
        if epoch % 10 == 0:
            print(f"{epoch}: Points reached: {points} - epsilon: {epsilon} - Best: {best_so_far}")

        model.save('output')


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