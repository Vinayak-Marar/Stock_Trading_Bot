from environment import Environment
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np

#CONSTANTS
NUM_EPISODE = 10000


env = Environment()
agent = Agent()

portfolio_values = []

def standard_scale(array: np.ndarray):
    mean = np.mean(array)
    std = np.std(array)

    return (array - mean)/std

def play_one_episode(agent, env):
    s =env.reset()
    s = standard_scale(s)

    done = False
    while not done:
        a = agent.get_action(s)
        next_state, r, done, info = env.step(a)
        next_state = standard_scale(next_state)
        
        if train_mode:
            agent.train(s, a, r, next_state, done)

        s = next_state

train_mode = True
for _ in range(NUM_EPISODE):
    val = play_one_episode(agent, env)
    portfolio_values.append(val)

plt.plot(portfolio_values)
plt.title("Portfolio Values")
plt.show()