#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
np.random.seed(seed=1)
x = np.linspace(0, 20, 100)
plt.plot(x, np.sin(x))
plt.show() 


###create dataset
nb_of_samples = 20
sequence_len = 10
X = np.zeros((nb_of_samples,sequence_len))
for row_idx in range(nb_of_samples):
    X[row_idx,:] = np.around(np.random.rand(sequence_len)).astype(int)
t = np.sum(X,axis=1)

print(X)

##define forward step functions
def update_state(xk,sk,wx,wRec):
    return xk*wx + sk*wRec

def forward_states(X,wx,wRec):
    S = np.zeros((X.shape[0],X.shape[1]+1))
    for k in range(0,X.shape[1]):
        S[:,k+1] = update_state(X[:,k],S[:,k],wx,wRec)
    return S

def loss(y,t):
    return np.mean((t-y)**2)





#%%
import numpy as np
A = np.array([[12,3],[23,4],[2,4],[99,234]])
print(A[range(A.shape[0]),[1,1,1,0]])

