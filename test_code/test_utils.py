import random

ENG_FAIL_PROBAB = 0.2

def engine_fail_probab(action):
    if random.random() < ENG_FAIL_PROBAB:
        return 0
    else:
        return action

def if_engine_fail(ep, num_ep_fail):
    if (ep+1)%(num_ep_fail+1) == 0:
        return True