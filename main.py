import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

def minimum_cost(k):
    seed_value = 5
    values = []
    for i in range(0, seed_value):
        M, P = common.init(X, k, seed=i)
        mixture, post, cost = kmeans.run(X, M, P)
        values.append(cost)
    min_cost = min(values)
    min_seed = values.index(min_cost)

    M, P = common.init(X, k, seed=min_seed)
    mixture, post, cost = kmeans.run(X, M, P)
    print("The minimum cost of K =", k, "is", cost)
    common.plot(X, mixture, post, title=f'K = {k}')

# TODO ONE Cluster (K=1)
minimum_cost(1)
# TODO TWO Clusters (K=2)
minimum_cost(2)
# TODO THREE Clusters (K=3)
minimum_cost(3)
# TODO FOUR Clusters (K=4)
minimum_cost(4)