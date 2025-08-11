import numpy as np

class Linear:

    def __init__(self,n_dims, n_actions):
        self.w = np.random.randn(n_dims, n_actions)/np.sqrt(n_dims)
        self.b = np.zeros(n_actions)

        self.losses = []

    def predict(self, X):
        X = np.atleast_2d(X)
        return np.dot(X,self.w) + self.b
    
    def grad(self, X, y, lr=0.01):

        X = np.atleast_2d(X).astype(float)   # (batch, n_dims)
        y = np.atleast_2d(y).astype(float)  

        
        yhat = self.predict(X)
        errors = yhat - y

        gw = X.T.dot(errors) / X.shape[0]              # (n_dims, n_actions)
        gb = errors.mean(axis=0)   
        
        self.w -= lr*gw
        self.b -= lr*gb

        mse = np.mean((errors)**2)
        self.losses.append(mse)

    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.w = npz['W']
        self.b = npz['b']

    def save_weights(self, filepath):
        np.savez(filepath, W= self.w, b=self.b)