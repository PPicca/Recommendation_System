import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")
K_1 = 1
M, P = common.init(X, K_1, seed = 4)
mixture, post, cost = kmeans.run(X, M, P)
print("The cost of K =", K_1, "is", cost)
common.plot(X, mixture, post, title = 'K = 1')
