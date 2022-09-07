import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym


env_name = 'CartPole-v1'
env = gym.make(env_name)
observation = env.reset()
model = keras.models.load_model('output')

for counter in range(2000):
    env.render()

    # TODO: Get discretized observation
    action = np.argmax(model.predict(observation.reshape([1, 1, 4])))

    # TODO: Perform the action
    observation, reward, done, info = env.step(action)  # Finally perform the action

    if done:
        print(f"done")
        break
env.close()

