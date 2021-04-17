import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")
K_1 = 1
M, P = common.init(X, K_1, seed = 0)
mixture, post, cost = kmeans.run(X, M, P)
print("The cost of K =", K_1, "is", cost)
common.plot(X, mixture, post, title = 'K = 1')

K_2 = 2
M, P = common.init(X, K_2, seed = 0)
mixture, post, cost = kmeans.run(X, M, P)
print("The cost of K =", K_2, "is", cost)
common.plot(X, mixture, post, title = 'K = 2')

K_3 = 3
M, P = common.init(X, K_3, seed = 0)
mixture, post, cost = kmeans.run(X, M, P)
print("The cost of K =", K_3, "is", cost)
common.plot(X, mixture, post, title = 'K = 3')

K_4 = 4
M, P = common.init(X, K_4, seed = 0)
mixture, post, cost = kmeans.run(X, M, P)
print("The cost of K =", K_4, "is", cost)
common.plot(X, mixture, post, title = 'K = 4')