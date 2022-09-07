import time
import gym

# import necessary blocks from keras to build the Deep Learning backbone of our agent
from tensorflow.keras.models import Sequential  # To compose multiple Layers
from tensorflow.keras.layers import Dense  # Fully-Connected layer
from tensorflow.keras.layers import Activation  # Activation functions
from tensorflow.keras.layers import Flatten  # Flatten function
from tensorflow.keras.optimizers import Adam  # Adam optimizer
from rl.memory import SequentialMemory  # Sequential Memory for storing observations ( optimized circular buffer)
from rl.agents.dqn import DQNAgent  # Use the basic Deep-Q-Network agent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


def create_env():
    env_name = 'CartPole-v1'
    env = gym.make(env_name, render_mode='human')
    return env


def create_nn(env):
    nb_actions = env.action_space.n  # get the number of possible actions

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))

    model.add(Dense(16))
    model.add(Activation('relu'))

    model.add(Dense(32))
    model.add(Activation('relu'))

    model.add(Dense(32))
    model.add(Activation('relu'))

    model.add(Dense(nb_actions))  # Final output layer of two actions
    model.add(Activation('linear'))

    return model


def run_environment(command):
    env = create_env()
    model = create_nn(env)
    memory = SequentialMemory(limit=20000, window_length=1)  # experience replay memory
    nb_actions = env.action_space.n

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                                  attr='eps',
                                  value_max=1.,
                                  value_min=.1,
                                  value_test=.05,
                                  nb_steps=30000)

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=100, policy=policy)

    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])  # Mean absolute error for metrics

    if command == 'train':
        dqn.fit(env, nb_steps=30000, visualize=False, verbose=2)

        dqn.save_weights('dqn_CartPole_v0_weights.h5f', overwrite=True)
    elif command == 'test':
        dqn.load_weights('dqn_CartPole_v0_weights.h5f')
        dqn.test(env, nb_episodes=5, visualize=True)
    else:
        print("command not recognised, closing")

    env.close()


run_environment('test')
