# **********************THIS FILE IS JUST CREATED FOR TESTING PURPOSE**********************

import gymnasium as gym
import numpy as np

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to generate the first observation
# observation, info = env.reset(seed=42)
state = env.reset()
# print(type(state))
# state_qn = np.expand_dims(state[0], axis=0)
# print(state_qn, end="\n")
# next_state, *_ = env.step(1)
# print(next_state, end="\n")


# state = next_state.copy()
print(type(state[0]))

for _ in range(10):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    # observation, reward, terminated, truncated, info = env.step(action)
    # print(f"Observation={observation}\nReward={reward}\nInfo={info}")
    # print("\n\n")
    # print(observation.shape)
    # print("____________________________________________________________________________________")

    # If the episode has ended then we can reset to start a new episode
    # if terminated or truncated:
    #     observation, info = env.reset()

env.close()