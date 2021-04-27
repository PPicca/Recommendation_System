import numpy as np
import em
import common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0

seed_value = 5
values = []
for i in range(0, seed_value):
    M_1, P_1 = common.init(X, K, seed=i)
    mix, pT, l = em.run(X, M_1, P_1)
    values.append(l)
max_ll = max(values)
max_seed = values.index(max_ll)

M, P = common.init(X, K, seed=max_seed)
mixture, post, ll = em.run(X, M, P)

X_pred = em.fill_matrix(X, mixture)
rmse = common.rmse(X_gold, X_pred)

if rmse == 0.3152301205749675:
    print("PASS")
else:
    print("FAIL")