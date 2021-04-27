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

X_incomplete = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt('netflix_complete.txt')

def max_likelihood_incomplete_data(k):
    seed_value = 5
    values = []
    for i in range(0, seed_value):
        M, P = common.init(X_incomplete, k, seed=i)
        mixture, post, ll = em.run(X_incomplete, M, P)
        values.append(ll)
    max_ll = max(values)
    max_seed = values.index(max_ll)

    M, P = common.init(X_incomplete, k, seed=max_seed)
    mixture, post, ll = em.run(X_incomplete, M, P)
    print("The maximum likelihood of K =", k, "is", ll)
    return mixture, ll, k

# ONE Cluster (K = 1)
mx, li, k = max_likelihood_incomplete_data(1)

# TWELVE Clusters (K = 12)
mx, li, k = max_likelihood_incomplete_data(12)

# MATRIX COMPARISON WITH 12 CLUSTERS (PREDICTED VS COMPLETE)
X_pred = em.fill_matrix(X_incomplete, mx)
rmse = common.rmse(X_gold, X_pred)
print(rmse)