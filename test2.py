import os
from datetime import datetime
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from environment import Environment
from agent import Agent, play_one_episode

# --------------------- CONFIG ---------------------
DATA_PATH = "data.csv"
WEIGHTS_PATH = "model/model_ep100.npz"  # must exist from training
SCALER_PATH = "model/scaler.pkl"        # must exist from training
INITIAL_INVESTMENT = 200000.0
OUT_DIR = "models"
# --------------------------------------------------

def maybe_make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def run_test(data_path=DATA_PATH, weights_path=WEIGHTS_PATH,
             scaler_path=SCALER_PATH,
             initial_investment=INITIAL_INVESTMENT, out_dir=OUT_DIR):
    
    # Load data
    data = pd.read_csv(data_path)
    stock_data = data.drop(["Date"], axis=1)
    n_time_steps, num_shares = stock_data.shape
    train_steps = n_time_steps // 2
    test_data = stock_data[train_steps:]

    state_size = 2 * num_shares + 1
    action_size = 3 ** num_shares

    env = Environment(test_data, initial_investment)
    agent = Agent(state_size=state_size, action_size=action_size)

    # Load scaler
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    print(f"Loaded scaler from: {scaler_path}")

    # Load model weights
    if os.path.exists(weights_path):
        try:
            agent.load_weights(weights_path)
            print(f"Loaded weights from: {weights_path}")
        except Exception as e:
            print(f"Failed to load weights from {weights_path}: {e}\nContinuing with random weights.")
    else:
        print(f"Weights file not found at {weights_path}. Running with random weights.")

    # Greedy policy for testing
    agent.epsilon = 0.0

    # Run episode
    t0 = datetime.now()
    final_portfolio = play_one_episode(agent, env, train_mode=False, scaler=scaler)
    dt = datetime.now() - t0

    print(f"Test finished — final portfolio value: {final_portfolio:.2f}, duration: {dt}")

    # Save results
    maybe_make_dir(out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(out_dir, f"test_portfolio_value_{timestamp}.csv")
    pd.Series([final_portfolio], name="portfolio_value").to_csv(out_csv, index=False)
    print(f"Saved test result to: {out_csv}")

    return final_portfolio

if __name__ == "__main__":
    values = []
    for i in range(10,3340,10):
        val = run_test(weights_path=f"model/model_ep{i}.npz")
        values.append(val)

    plt.plot(values)
    plt.title("Values")
    plt.show()