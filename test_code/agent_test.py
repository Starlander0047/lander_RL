import tensorflow as tf
import numpy as np
import gymnasium as gym
import random
import test_utils

ll_model = tf.keras.models.load_model("/home/starlander0047/gymtest/lunar_lander_solved[-9.8=220=128N=128MBS].keras")
# ll_model.summary()

env = gym.make("LunarLander-v3", render_mode="human", gravity=-9.8)
# enable_wind=True, wind_power=20.0, turbulence_power=1.9

# Variables Start
num_episodes=15
num_steps=1000
# Variables End


for ep in range(num_episodes):
    observation, info = env.reset()
    # observation = test_utils.addNoiseXY(observation)  # Adding coordinate noise
    total_rewards=0

    for t in range(num_steps):
        
        returns = ll_model(np.expand_dims(observation, axis=0))
        action = np.argmax(returns)
        # action = np.random.choice([0,1,2,3], 1)[0]

        if test_utils.engineFailure():   # Engine Failure Code
            action = 0
    

        observation, reward, terminated, truncated, info = env.step(action)
        observation = test_utils.addNoiseXY(observation)  # Adding coordinate noise

        total_rewards+=reward

        # print(f"Observation={observation}\nReward={reward}\nInfo={info}")
        # print("\n\n")
        # print("____________________________________________________________________________________")

        if terminated or truncated:
            break
    print(f"Total Rewards For Last Episode = {total_rewards}")

env.close()
