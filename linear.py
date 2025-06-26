import numpy as np 


class Nn:
    def __init__(self):
        np.random.seed(1)
        self.w = 2*np.random.rand(2,1) - 1
        
        # print(self.w.shape)
        self.b = np.random.rand(1)

    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    def backward(self, x, y, lr=0.01):
        
        z = np.dot(x, self.w) + self.b 
        
        error = y - z
        # print(error,y.shape)
        
        d_a = error 
        
        # print(self.w.shape,np.dot(x.T, d_a).shape,d_a.shape)
        self.w += np.dot(x.T, d_a) * lr
        # print(self.w.shape)
        self.b += np.sum(d_a, axis=0) * lr
    
    def train(self,x,y,ep):
        for _ in range(ep):
            res=self.forward(x)
            # print(f"loop {_} output ",res)
            self.backward(x,y)

    def predict(self,x):
        return self.forward(x)

inp = np.array([[1,0],[0,1],[1,1],[0,0]])

out = np.array([[1],[1],[2],[0]])
nn = Nn()

nn.train(inp,out,300)

print(nn.predict(inp))