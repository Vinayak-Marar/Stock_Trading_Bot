from environment import Environment
from agent import Agent,play_one_episode,get_scaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime
import joblib

#CONSTANTS
NUM_EPISODE = 10000
INITIAL_INVESTMENT = 200000
MODEL = 'model'
REWARDS = 'rewards'

def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)


data = pd.read_csv("data.csv")
stock_data = data.drop(["Date"], axis=1)

n_time_steps, num_shares = stock_data.shape

train_steps = n_time_steps //2
state_size = 2*num_shares + 1
action_size = 3 ** num_shares

train_data = stock_data[: train_steps]
test_data = stock_data[train_steps: ]

def train():

    maybe_make_dir(MODEL)
    maybe_make_dir(REWARDS)

    env = Environment(train_data, INITIAL_INVESTMENT)
    agent = Agent(state_size=state_size, action_size=action_size)
    scaler = get_scaler(env)
    os.makedirs(MODEL, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL, "scaler.pkl"))

    portfolio_values = []

    train_mode = True
    for e in range(NUM_EPISODE):
        t0 = datetime.now()
        val = play_one_episode(agent, env,scaler, train_mode)

        dt = datetime.now() - t0

        print(f"episode: {e + 1}/{NUM_EPISODE}, episode end value: {val:.2f}, duration: {dt}")
        portfolio_values.append(val)

        if (e + 1) % 10 == 0:
            filename = os.path.join(MODEL, f"model_ep{e+1}.npz")
            agent.save_weights(filename)
            print(f"Saved checkpoint: {filename}")

    # final save
    final_path = os.path.join(MODEL, "model_final.npz")
    agent.save_weights(final_path)
    print(f"Saved final model: {final_path}")

    results_csv = os.path.join(MODEL, "train_portfolio_values.csv")
    pd.Series(portfolio_values).to_csv(results_csv, index=False)
    print(f"Saved training portfolio values to: {results_csv}")

    plt.plot(portfolio_values)
    plt.title("Portfolio Values")
    plt.xlabel("Episode")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
   train()