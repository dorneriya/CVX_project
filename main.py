import numpy as np
from pandas import read_pickle

EXAMPLES_PATH = 'examples.pkl'
EXAMPLES2_PATH = 'examples2.pkl'
MAX_ITERS = 20000
DEBUG_MODE = False


class Example:
    def __init__(self, S: np.ndarray, K: np.ndarray, k: int):
        self.k = k
        self.S = S
        self.K = K


def ones_k_band_lower(n: int, k: int):
    return np.triu(np.tril(np.ones((n, n)), 0), -k)


def target(S: np.ndarray, K: np.ndarray) -> float:
    """
    Original target function for minimization
    :param S: The given S matrix
    :param K: a k-banded pd matrix
    :return: the target value (the one we should minimize)
    """
    return np.trace(S @ K) - np.log(np.linalg.det(K))


def triangular_target(S: np.ndarray, L: np.ndarray) -> float:
    """
    Returns the target value for the L triangular matrix (where K = L @ L.T)
    :param S: The given S matrix
    :param L: Our triangular L k-band matrix
    :return: the target value
    """
    return (L.T * (S @ L).T).sum() - np.sum(np.log(np.power(np.diag(L), 2)))


def target_gradient(S: np.ndarray, L: np.ndarray) -> float:
    """
    C alcultes the gradient of the target problem (in it's cholesky form)
    :param S: The given S matrix
    :param L: Our triangular L k-band matrix
    :return: the target gradient value
    """
    log_mat = np.diag(1 / (np.diag(L)))
    return 2 * (S @ L - log_mat)


def solve(S: np.ndarray, k: int):
    """
    The main function of the program - this will be the function that the tests run
    :param S: The given S matrix
    :param k: The k-band parameter
    :return: A k-banded pd matrix K that minimizes the target function
    """
    return solve_problem_with_parameters(S, k, alpha=0.4, beta=0.5)


def solve_problem_with_parameters(S, k, alpha, beta):
    """
    Gives a solution to the target problem with given parameters for the gradient decent
    :param S: The given S matrix
    :param k: The k-band parameter
    :param alpha: the alpha parameter for the gradient decent method
    :param beta: the beta parameter for the gradient decent method
    :return:
    """
    sub_L_array = []
    sub_problem_size = k + 1
    n = S.shape[0]
    for i in range(0, n):
        if i + k >= n:
            break
        sub_S = S[i: min(n, i + sub_problem_size), i: min(n, i + sub_problem_size)]
        sub_L = solve_helper(sub_S, k, alpha, beta)
        sub_L_array.append(sub_L)

    return concat_sub_L_array(sub_L_array, sub_problem_size, n)


def concat_sub_L_array(sub_L_array, size, n):
    """
    Gets an array of sub_solution matrix's and concat them to one K solution -
    The sub-solutions are k-banded and pd and so is the concatenated K.
    :param sub_L_array:
    :param size:
    :param n:
    :return: K
    """
    concatenated_L = np.zeros((n, n))
    for i, sub_L in enumerate(sub_L_array):
        concatenated_L[i: min(n, i + size), i: min(n, i + size)] = sub_L

    return concatenated_L @ concatenated_L.T


def solve_helper(sub_S: np.ndarray, k: int, alpha: float, beta: float) -> np.ndarray:
    """
    Solves each given sub-problem using initial guess and gradient descent.
    :param sub_S: The given S matrix
    :param k: The k-band parameter
    :param alpha: the alpha parameter for the gradient decent method
    :param beta: the beta parameter for the gradient decent method
    :return: a triangular L matrix such that L @ L.T is the optimal K matrix (for the sub matrix of S)
    """
    try:
        # Best guess if sub_S is non-singular
        initial_L = np.linalg.cholesky(np.linalg.inv(sub_S)) * ones_k_band_lower(sub_S.shape[0], k)
    except:
        # Good guess if sub_S is singular
        initial_L = (1 / sub_S) * ones_k_band_lower(sub_S.shape[0], k)

    return gradient_decent(sub_S, initial_L, k, alpha, beta)


def gradient_decent(sub_S: np.ndarray, sub_L: np.ndarray, k: int, alpha: float, beta: float, max_iterations=MAX_ITERS):
    """
    Tries to find global minimum with a gradient decent implementation
    :param sub_S: The given sub_S
    :param sub_L: The triangular pd k-band sub_L s.t sub_L @ sub_L.T = sub_K
    :param k: k-band parameter
    :param alpha: determines how much the norma of the gradient will get considered
    :param beta:  determines how fast the step-size will reduce
    :param max_iterations: maximal number of iterations we will run
    :return: approximation for the arg min of the sub-problem
    """
    ones_k_band_lower_mat = ones_k_band_lower(sub_S.shape[0], k)
    threshold = 0.001
    prev_value = triangular_target(sub_S, sub_L)
    for i in range(max_iterations):
        grad = target_gradient(sub_S, sub_L) * ones_k_band_lower_mat
        step_size = line_search(triangular_target, sub_S, sub_L, grad, alpha, beta)
        sub_L -= step_size * grad
        if i % 100 == 0 or i == 1:
            current_value = triangular_target(sub_S, sub_L)
            if abs(current_value - prev_value) < threshold:
                if DEBUG_MODE:
                    print(f'number of iterations was {i + 1}')
                break
            prev_value = current_value

    else:
        if DEBUG_MODE:
            print(f'number of iterations was {max_iterations}')

    return sub_L


def line_search(f, S, L, grad, alpha, beta):
    t = 1
    cur_target_val = f(S, L)
    grad_norm = np.linalg.norm(grad) ** 2
    while f(S, L - t * grad) > cur_target_val - alpha * t * grad_norm:
        t *= beta
    return t


def run_solve_on_given_exmaples(examples_path):
    examples = read_pickle(examples_path)
    for i in range(len(examples)):
        S, K, k = examples[i].S, examples[i].K, examples[i].k
        our_K = solve(S, k)
        print(f"error {i}: {np.linalg.norm(our_K - K) / (k * S.shape[0])}")
        print("Opt: ", target(S, K))
        print("Our: ", target(S, our_K))


if __name__ == '__main__':
    run_solve_on_given_exmaples(EXAMPLES2_PATH)
