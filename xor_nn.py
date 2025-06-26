import numpy as np

def sig(x):
    return 1/(1+np.exp(-x))

def d_sig(z):
    return z*(1-z)

class Neuron:
    def __init__(self):
        np.random.seed(4)  
        self.w1 = 2*np.random.rand(2,2)-1
        self.b1 = np.zeros((1,2))
        self.w2 = 2*np.random.rand(2,1)-1
        self.b2 = np.zeros((1,1))
    
    def train(self, x, y, ep, lr=0.1):
        for _ in range(ep):
           
            a = x.dot(self.w1) + self.b1
            h = sig(a)
            b = h.dot(self.w2) + self.b2
            z = sig(b)

           
            e2 = z - y
            d_e2 = e2 * d_sig(z)

            e1 = d_e2.dot(self.w2.T)
            d_e1 = e1 * d_sig(h)

            
            self.w2 -= h.T.dot(d_e2) * lr
            self.w1 -= x.T.dot(d_e1) * lr
            self.b2 -= np.sum(d_e2, axis=0, keepdims=True) * lr
            self.b1 -= np.sum(d_e1, axis=0, keepdims=True) * lr

    def think(self, x):
        a = x.dot(self.w1) + self.b1
        h = sig(a)
        b = h.dot(self.w2) + self.b2
        z = sig(b)
        return z

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

nn = Neuron()
nn.train(x, y, 10000, 0.1)

print((nn.think(x)))
