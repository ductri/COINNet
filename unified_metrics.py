import numpy as np
from sklearn import metrics
from scipy.sparse import csc_matrix
from scipy.optimize import linear_sum_assignment


def evaluate(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    acc = ((pred == label)*1.0).mean()
    return nmi, ari, acc


def match_it(M1, M2):
    assert M1.shape[0] < M1.shape[1]

    M1 = M1/(np.sqrt((M1*M1).sum(1, keepdims=True))+1e-9*np.ones(M2.shape[1]))
    M2 = M2/(np.sqrt((M2*M2).sum(1, keepdims=True))+1e-9*np.ones(M2.shape[1]))

    K = M1.shape[0]
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            tmp = M1[i, :] - M2[j, :]
            cost[i, j] = (np.power(tmp, 2)).sum()
    _, best_ind = linear_sum_assignment(cost)
    new_M2 = M2[best_ind, :]
    PI = np.zeros((K, K))
    for i in range(K):
        PI[i, best_ind[i]] = 1

    return new_M2, best_ind, PI, M1

def match_it_label_with_PI(Y1, Y2, K):
    """Contains indices
    Return a modified Y2
    """
    N = Y1.shape[0]
    Y1_onehot = csc_matrix((np.ones(N), (Y1, range(N))), shape=(K, N)).toarray()
    Y2_onehot = csc_matrix((np.ones(N), (Y2, range(N))), shape=(K, N)).toarray()
    _, _, PI, _ = match_it(Y1_onehot, Y2_onehot)
    Y2_onehot_ = PI@Y2_onehot
    return Y2_onehot_.argmax(0), PI


def match_it_label(Y1, Y2, K):
    """Contains indices
    Return a modified Y2
    """
    N = Y1.shape[0]
    Y1_onehot = csc_matrix((np.ones(N), (Y1, range(N))), shape=(K, N)).toarray()
    Y2_onehot = csc_matrix((np.ones(N), (Y2, range(N))), shape=(K, N)).toarray()
    _, _, PI, _ = match_it(Y1_onehot, Y2_onehot)
    Y2_onehot_ = PI@Y2_onehot
    return Y2_onehot_.argmax(0)


def MSE(M1, M2):
    new_M2, _, _, new_M1 = match_it(M1, M2)
    K = M1.shape[0]
    rel = new_M1 - new_M2
    return (rel*rel).sum()/K


def best_acc(label, pred, K):
    """
    Evaluate best acc respect to label permutation
    """
    pred = match_it_label(label, pred, K)
    return np.mean((pred == label)*1.0)



if __name__ == "__main__":
    K = 10
    Y1 = np.random.randint(0, K, size=(10000))
    Y2 = np.random.randint(0, K, size=(10000))
    Y2 = match_it_label(Y1, Y2, K)
    nmi, ari, acc = evaluate(Y1, Y2)
    print(f'NMI: {nmi} ARI: {ari} ACC: {acc}')

    Y2 = np.zeros(Y1.shape)
    for i in range(9):
        Y2[Y1==i] = i+1
    Y2[Y1==9] = 0

    Y2 = match_it_label(Y1, Y2, K)
    nmi, ari, acc = evaluate(Y1, Y2)
    print(f'NMI: {nmi} ARI: {ari} ACC: {acc}')

    K = 3
    N = 1000
    M = np.zeros((K, N))
    M[np.random.randint(0, 3, N), np.arange(N)] = 1
    M = M + 0.01*np.random.rand(K, N)
    M[M<0] = 0
    M = M/M.sum(0)
    M1 = M

    M = np.zeros((K, N))
    M[np.random.randint(0, 3, N), np.arange(N)] = 1
    M = M + 0.01*np.random.rand(K, N)
    M[M<0] = 0
    M = M/M.sum(0)
    M2 = M
    print(MSE(M1, M2))
