import numpy as np
import tensorflow as tf
import random #-------------------------------------------------------------------------------May give a SEED

import base64
from itertools import zip_longest
import imageio
import IPython
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from statsmodels.iolib.table import SimpleTable


E_MIN = 0.01
MINIBATCH_SIZE = 128
E_DECAY = 0.995
TAU = 1e-3
ENGINE_FAIL_PROBAB = 0.05

def get_new_eps(epsilon):
    return max(E_MIN, E_DECAY*epsilon)

def get_action(q_values, epsilon=0):
    if random.random() > epsilon:
        return np.argmax(q_values.numpy()[0]) # Exploitation
    else:
        return random.choice(np.arange(4))

def check_update_condition(t, NUM_STEPS_FOR_UPDATE, memory_buffer):
    if len(memory_buffer) > MINIBATCH_SIZE and (t+1)%NUM_STEPS_FOR_UPDATE == 0:
        return True
    else:
        return False

def get_experiences(memory_buffer):
    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)

    states = tf.convert_to_tensor( np.array([e.state for e in experiences if e is not None]), dtype=tf.float32 )
    actions = tf.convert_to_tensor( np.array([e.action for e in experiences if e is not None]), dtype=tf.float32 )
    rewards = tf.convert_to_tensor( np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32 )
    next_states = tf.convert_to_tensor( np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32 )
    terminateds = tf.convert_to_tensor(np.array([e.terminated for e in experiences if e is not None]).astype(np.uint8), dtype=tf.float32)

    return (states, actions, rewards, next_states, terminateds)
    

def update_target_network(q_network, target_q_network):
    for target_net_weight , q_net_weight in zip(target_q_network.weights, q_network.weights):
        target_net_weight.assign(TAU*q_net_weight + (1-TAU)*target_net_weight)


def compute_loss(experiences, GAMMA, q_network, target_q_network):
    states, actions, rewards, next_states, terminated_vals = experiences
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
    
    y_targets = rewards + GAMMA*max_qsa*(1-terminated_vals)

    q_values = q_network(states)
    q_values = tf.gather_nd( q_values, tf.stack([tf.range(q_values.shape[0]), tf.cast(actions, tf.int32)], axis=1) )

    loss = tf.keras.losses.MSE(y_targets, q_values)
    return loss

@tf.function
def agent_learn(experiences, GAMMA, q_network, target_q_network, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, GAMMA, q_network, target_q_network) #---------------------------------------------------What is the type of loss returned and how is it getting differentiated?

    gradients = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    update_target_network(q_network, target_q_network)


# Functions created by me for adding noise
def addNoiseXY(state):
    noise = np.random.normal(0 ,0.02, 5)
    noiseVal = np.random.choice(noise, 1)
    state[0]+=noiseVal
    state[1]+=noiseVal

    return state

def engineFailure():
    if random.random() <= ENGINE_FAIL_PROBAB:
        return True
    else:
        return False


# Functions for Graph Plotting[Copy Pasted]
def plot_history(reward_history, rolling_window=20, lower_limit=None, upper_limit=None, plot_rw=True, plot_rm=True):

    if lower_limit is None or upper_limit is None:
        rh = reward_history
        xs = [x for x in range(len(reward_history))]
    else:
        rh = reward_history[lower_limit:upper_limit]
        xs = [x for x in range(lower_limit,upper_limit)]
    
    df = pd.DataFrame(rh)
    rollingMean = df.rolling(rolling_window).mean()

    plt.figure(figsize=(10,7), facecolor='white')
    
    if plot_rw:
        plt.plot(xs, rh, linewidth=1, color='cyan')
    if plot_rm:
        plt.plot(xs, rollingMean, linewidth=2, color='magenta')

    text_color = 'black'
        
    ax = plt.gca()
    ax.set_facecolor('black')
    plt.grid()
    #plt.title("Total Point History", color=text_color, fontsize=40)
    plt.xlabel('Episode', color=text_color, fontsize=30)
    plt.ylabel('Total Points', color=text_color, fontsize=30)
    yNumFmt = mticker.StrMethodFormatter('{x:,}')
    ax.yaxis.set_major_formatter(yNumFmt)
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    plt.show()