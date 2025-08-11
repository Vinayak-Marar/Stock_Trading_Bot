import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")
stock_data = data.drop(["Date"], axis=1)


class Environment:


    def __init__(self, initial_investment):
        self.pointer = 0
        self.initial_investment = initial_investment
        self.stock_prices = np.array(stock_data.iloc[self.pointer])
        self.portfolio_val = initial_investment
        self.cash = initial_investment
        self.num_shares = [0,0,0]

    def reset(self):
        self.pointer = 0
        self.stock_prices = np.array(stock_data.iloc[self.pointer])
        self.portfolio_val = self.initial_investment
        self.cash = self.initial_investment

    def step(self, action):
        self.pointer += 1
        current_stock_prices = np.array(stock_data.iloc[self.pointer])
        reward_per_share = current_stock_prices - self.stock_prices
        total_reward_per_share = reward_per_share * self.num_shares
        sum_reward = np.sum(total_reward_per_share)
        self.stock_prices = current_stock_prices

        self.trade(action)

        portfolio_val = np.dot(self.stock_prices, self.num_shares) + self.cash
        next_state = np.concatenate(self.stock_prices, self.num_shares)
        next_state.append(portfolio_val)

        return next_state, sum_reward, self.pointer==50, portfolio_val
    
    def trade(self, action): # eg. action = [0,1,2] where 0 = hold, 1 = buy, 2 = sell
        
        for i, a in enumerate(action): # sell completely before purchasing
            if a == 2:
                shares = self.num_shares[i]
                price_per_share = self.stock_prices[i]

                total_sell_price = shares * price_per_share
                self.cash += total_sell_price
                self.num_shares[i] == 0

        while True: # buy as much as possible
            if self.cash >= np.min(self.stock_prices): # to check cash availabilty

                for i,a in enumerate(action):
                    if a == 1 :
                        self.num_shares[i] += 1
                        self.cash -= self.stock_prices[i]
            else:
                break


                

