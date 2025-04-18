import gymnasium as gym
import numpy as np
import tensorflow as tf
import time
from collections import deque, namedtuple
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
import utils
import logging

#Logger Code Starts
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)  # Set to lowest level to capture everything
# Create handlers
reward_handler = logging.FileHandler('reward.log')
reward_handler.setLevel(logging.DEBUG)

# Create formatters
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Add formatters to handlers
reward_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(reward_handler)
#Logger Code Ends

# Environment Information
GRAVITY=-9.8
env = gym.make("LunarLander-v3", render_mode="human", gravity=GRAVITY)
state_size = env.observation_space.shape
num_actions = env.action_space.n
MEMORY_SIZE = 100_000     # size of memory buffer
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate  
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps
MIN_POINTS_TO_SOLVE = 220.0

observation, info = env.reset()


q_network = Sequential([
    Input(shape=state_size),
    Dense(units=128, activation="relu"),
    Dense(units=128, activation="relu"),
    Dense(units=num_actions, activation="linear")]
)
target_q_network = Sequential([
    Input(shape=state_size),
    Dense(units=128, activation="relu"),
    Dense(units=128, activation="relu"),
    Dense(units=num_actions, activation="linear")]
)
optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA)

start = time.time()
num_episodes = 2500
max_num_timesteps = 1500
total_point_history = []
num_p_av = 100
epsilon = 1.0
memory_buffer = deque(maxlen=MEMORY_SIZE)
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "terminated"])
target_q_network.set_weights(q_network.get_weights())

logger.critical(f"---------***********---------Gravity = {GRAVITY}, Min_PointsToSolve={MIN_POINTS_TO_SOLVE}")
for i in range(num_episodes):
    state = env.reset()[0]  # Here 'state' is np array
    total_points = 0
    state = utils.addNoiseXY(state)   # Added by me for noise addition to X and Y coordinates

    for t in range(max_num_timesteps):

        state_qn = np.expand_dims(state, axis=0)
        q_values = q_network(state_qn)
        action = utils.get_action(q_values, epsilon)

        # Code for Engine Failure starts
        if utils.engineFailure():
            action = 0
        # Code for Engine Failure ends

        next_state, reward, terminated, *_ = env.step(action)  # next_state is nd numpy array (8,)

        next_state = utils.addNoiseXY(next_state)   # Added by me for noise addition to X and Y coordinates
        memory_buffer.append(experience(state, action, reward, next_state, terminated))

        update = utils.check_update_condition(t, NUM_STEPS_FOR_UPDATE, memory_buffer) #----------------Why not just send the size of memory_buffer?

        if update:
            experiences = utils.get_experiences(memory_buffer)
            utils.agent_learn(experiences, GAMMA, q_network, target_q_network, optimizer)
        
        state = next_state.copy()
        total_points += reward

        if terminated:
            break
    
    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])

    epsilon = utils.get_new_eps(epsilon)

    print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} Episodes: {av_latest_points:.2f}", end="")
    logger.debug(f"Episode {i+1} received a Reward = {total_points} ")

    if (i+1) % num_p_av == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} Episodes: {av_latest_points:.2f}")
    
    if av_latest_points >= MIN_POINTS_TO_SOLVE:
        print(f"\n\nEnvironment Solved in {i+1} Episodes!")
        q_network.save("lunar_lander_solved[-9.8=220=128N=128MBS=0.05EF=0M,0.02SD,5P].keras")
        break

total_time = time.time()-start
print(f"Total Runtime: {total_time:.2} Seconds OR {(total_time/60):.2f} Minutes")



    
env.close()

utils.plot_history(total_point_history)