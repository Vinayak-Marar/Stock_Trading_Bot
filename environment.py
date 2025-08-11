import pandas as pd
import numpy as np
import itertools


class Environment:
    

    def __init__(self, data, initial_investment):
        self.data = data
        self.pointer = 0
        self.initial_investment = initial_investment
        self.stock_prices = np.array(self.data.iloc[self.pointer])
        self.portfolio_val = initial_investment
        self.cash = initial_investment
        self.n_step = self.data.shape[0]
        self.num_shares = [0] * self.data.shape[1]
        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.data.shape[1])))

    def reset(self):
        self.pointer = 0
        self.stock_prices = np.array(self.data.iloc[self.pointer])
        self.num_shares = [0,0,0]
        self.portfolio_val = self.initial_investment
        self.cash = self.initial_investment

        state = [*self.stock_prices, *self.num_shares, self.cash]
        return state

    def step(self, action):
        self.pointer += 1
        current_stock_prices = np.array(self.data.iloc[self.pointer])
        reward_per_share = current_stock_prices - self.stock_prices
        total_reward_per_share = reward_per_share * self.num_shares
        sum_reward = np.sum(total_reward_per_share)
        self.stock_prices = current_stock_prices

        self.trade(action)

        portfolio_val = np.dot(self.stock_prices, self.num_shares) + self.cash
        next_state = [*self.stock_prices, *self.num_shares, self.cash]
  
        return next_state, sum_reward, self.pointer==self.data.shape[0]-1, portfolio_val
    
    def trade(self, action): # eg. action = [0,1,2] where 0 = hold, 1 = buy, 2 = sell
        
        action = self.action_list[action]

        for i, a in enumerate(action): # sell completely before purchasing
            if a == 2:
                shares = self.num_shares[i]
                price_per_share = self.stock_prices[i]

                total_sell_price = shares * price_per_share
                self.cash += total_sell_price
                self.num_shares[i] = 0

        while True:
            bought_any = False
            for i, a in enumerate(action):
                if a == 1 and self.cash >= self.stock_prices[i]:
                    self.num_shares[i] += 1
                    self.cash -= self.stock_prices[i]
                    bought_any = True
                    
            if not bought_any or self.cash < np.min(self.stock_prices):
                break


                

