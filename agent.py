import numpy as np
from linear import Linear
from sklearn.preprocessing import StandardScaler

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.model = Linear(state_size, action_size)

    def epsilon_greedy(self, state):
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            values = self.model.predict(state)
            return int(np.argmax(values[0]))

    def train(self, state, action, reward, next_state, done):
        if done == True:
            target = reward
        else:
            target = reward +  self.gamma * np.max(self.model.predict(next_state))

        target_full = self.model.predict(state)
        target_full[0,action] = target

        self.model.grad(state, target_full)

        if self.epsilon >= self.epsilon_min:
            self.epsilon *=self.epsilon_decay

    def load_weights(self, name):
        self.model.load_weights(name)

    def save_weights(self, name):
        self.model.save_weights(name)
        

def get_scaler(env):

  states = []
  for _ in range(env.n_step):
    action = np.random.choice(len(env.action_list))
    state, reward, done, info = env.step(action)
    states.append(state)
    if done:
      break

  scaler = StandardScaler()
  scaler.fit(states)
  return scaler



def play_one_episode(agent, env,scaler, train_mode):
    s =env.reset()
    s = np.array(s)
    s = scaler.transform([s])

    done = False
    while not done:
        a = agent.epsilon_greedy(s)

        # print(f"action {a} ")
        next_state, r, done, portfolio_val = env.step(a)
        next_state = scaler.transform([next_state])

        if train_mode:
            # print(f's {s}')
            # print(f'a {a}')
            # print(f'r {r}')
            # print(f'next _state {next_state}')

            agent.train(s, a, r, next_state, done)

        s = next_state
    return portfolio_val