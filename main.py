import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

def kmeans_minimum_cost(k):
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

# ONE Cluster (K=1)
kmeans_minimum_cost(1)

# TWO Clusters (K=2)
kmeans_minimum_cost(2)

# THREE Clusters (K=3)
kmeans_minimum_cost(3)

# FOUR Clusters (K=4)
kmeans_minimum_cost(4)

def maximum_likelihood(k):
    seed_value = 5
    values = []
    for i in range(0, seed_value):
        M, P = common.init(X, k, seed=i)
        mixture, post, ll = naive_em.run(X, M, P)
        values.append(ll)
    max_ll = max(values)
    max_seed = values.index(max_ll)

    M, P = common.init(X, k, seed=max_seed)
    mixture, post, ll = naive_em.run(X, M, P)
    print("The highest likelihood of K =", k, "is", ll)
    common.plot(X, mixture, post, title=f'K = {k}')
    return mixture, ll, k

# ONE Cluster (K=1)
mx, li, k = maximum_likelihood(1)
print(f"The BIC value for k = {k} is {common.bic(X, mx, li)}")
# TWO Clusters (K=2)
mx, li, k = maximum_likelihood(2)
print(f"The BIC value for k = {k} is {common.bic(X, mx, li)}")
# THREE Clusters (K=3)
mx, li, k = maximum_likelihood(3)
print(f"The BIC value for k = {k} is {common.bic(X, mx, li)}")
# FOUR Clusters (K=4)
mx, li, k = maximum_likelihood(4)
print(f"The BIC value for k = {k} is {common.bic(X, mx, li)}")