import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")


K_1 = 1
seed_value = 5
values = []
for i in range(0,seed_value):
    M, P = common.init(X, K_1, seed = i)
    mixture, post, cost = kmeans.run(X, M, P)
    values.append(cost)
min_cost = min(values)
min_seed = values.index(min_cost)
M, P = common.init(X, K_1, seed = min_seed)
mixture, post, cost = kmeans.run(X, M, P)
print("The minimum cost of K =", K_1, "is", cost)
common.plot(X, mixture, post, title = 'K = 1')

K_2 = 2
seed_value = 5
values = []
for i in range(0,seed_value):
    M, P = common.init(X, K_2, seed = i)
    mixture, post, cost = kmeans.run(X, M, P)
    values.append(cost)
min_cost = min(values)
min_seed = values.index(min_cost)
M, P = common.init(X, K_2, seed = min_seed)
mixture, post, cost = kmeans.run(X, M, P)
print("The minimum cost of K =", K_2, "is", cost)
common.plot(X, mixture, post, title = 'K = 2')

K_3 = 3
seed_value = 5
values = []
for i in range(0,seed_value):
    M, P = common.init(X, K_3, seed = i)
    mixture, post, cost = kmeans.run(X, M, P)
    values.append(cost)
min_cost = min(values)
min_seed = values.index(min_cost)
M, P = common.init(X, K_3, seed = min_seed)
mixture, post, cost = kmeans.run(X, M, P)
print("The minimum cost of K =", K_3, "is", cost)
common.plot(X, mixture, post, title = 'K = 3')

K_4 = 4
seed_value = 5
values = []
for i in range(0,seed_value):
    M, P = common.init(X, K_4, seed = i)
    mixture, post, cost = kmeans.run(X, M, P)
    values.append(cost)
min_cost = min(values)
min_seed = values.index(min_cost)
M, P = common.init(X, K_4, seed = min_seed)
mixture, post, cost = kmeans.run(X, M, P)
print("The minimum cost of K =", K_4, "is", cost)
common.plot(X, mixture, post, title = 'K = 4')