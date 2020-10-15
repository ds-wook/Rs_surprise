import numpy as np
from sklearn.metrics import mean_squared_error

def get_rmse(R, P, Q, non_zeros):
    error = 0
    full_pred_matrix = np.dot(P, Q.T)

    x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
    y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
    R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]
    full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]
    mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
    rmse = np.sqrt(mse)

    return rmse
def matrix_factorization(R, K, steps = 200, learning_rate = 0.01, r_lambda = 0.01):
    num_users, num_items = R.shape
    np.random.seed(1)
    P = np.random.normal(scale=1 / K, size = (num_users, K))
    Q = np.random.normal(scale=1 / K, size = (num_items, K))

    non_zeros = [(i, j, element) for i, vector in enumerate(R) for j, element in enumerate(vector) if element > 0]

    print("Learning Start!")
    for step in range(steps):
        for i, j, r in non_zeros:
            eij = r - np.dot(P[i, :], Q[j, :])
            P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
            Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])
        
        rmse = get_rmse(R, P, Q, non_zeros)

        

        if step % 10 == 0:
            print(f"### iteration step : {step}, rmse : {rmse:.3f}")

    print("Learning Finish")
    return P, Q