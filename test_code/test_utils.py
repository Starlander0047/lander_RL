import numpy as np
import tensorflow as tf
import random

ENGINE_FAIL_PROBAB = 0.05

# Functions created by me for adding noise
def addNoiseXY(state):
    noise = np.random.normal(0 ,0.4, 5)
    noiseVal = np.random.choice(noise, 1)
    state[0]+=noiseVal
    state[1]+=noiseVal

    return state

def engineFailure():
    if random.random() <= ENGINE_FAIL_PROBAB:
        return True
    else:
        return False