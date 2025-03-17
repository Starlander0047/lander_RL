import tensorflow as tf
import numpy as np
import gymnasium as gym
import test_utils
import random

ll_model = tf.keras.models.load_model("train_code/lunar_lander_solved[128].keras")
# ll_model.summary()

env = gym.make("LunarLander-v3", render_mode="human", gravity=-9.8)
# enable_wind=True, wind_power=20.0, turbulence_power=1.9

# Variables Start
num_episodes=10
num_steps=1000
# Variables End

# Code to insert engine failures starts
num_ep_fail=1   # can randomize it later
# Code to insert engine failures ends

for ep in range(num_episodes):
    observation, info = env.reset()
    total_rewards=0
    # engine_fail_ep = test_utils.if_engine_fail(ep, num_ep_fail)

    for t in range(num_steps):
        
        returns = ll_model(np.expand_dims(observation, axis=0))
        action = np.argmax(returns)

        # if engine_fail_ep and action==2:
        #     action = test_utils.engine_fail_probab(action)     # currently for main engine fail
            ##***** Engine failure generally shows malfunction or non active engine at some times

        observation, reward, terminated, truncated, info = env.step(action)
        total_rewards+=reward
        # print(f"Observation={observation}\nReward={reward}\nInfo={info}")
        # print("\n\n")
        # print("____________________________________________________________________________________")

        if terminated or truncated:
            break
    print(f"Total Rewards For Last Episode = {total_rewards}")

env.close()
