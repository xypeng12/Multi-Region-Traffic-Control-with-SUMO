import numpy as np
from qpsolvers import solve_qp


def make_positive_semidefinite(Q, epsilon=1e-6):
    eigenvalues = np.linalg.eigvals(Q)
    if np.all(eigenvalues >= 0):
        print("Q is already positive semidefinite.")
        return Q

    Q_new = Q + epsilon * np.eye(Q.shape[0])
    return Q_new
def add_slack_variables(Q, p, G, H, A, b, slack_weight=10):
    num_vars = Q.shape[0] 
    num_ineq_constraints = G.shape[0]  
    num_eq_constraints = A.shape[0]  

    new_Q = np.pad(Q, ((0, num_ineq_constraints), (0, num_ineq_constraints)), 'constant')
    new_p = np.pad(p, (0, num_ineq_constraints), 'constant')

    for i in range(num_ineq_constraints):
        new_Q[num_vars + i, num_vars + i] = 2.0 * slack_weight 

    new_G = np.pad(G, ((0, 0), (0, num_ineq_constraints)), 'constant')

    for i in range(num_ineq_constraints):
        new_G[i, num_vars + i] = 1.0 

    new_H = H.copy()

    new_A = np.pad(A, ((0, 0), (0, num_ineq_constraints)), 'constant')

    new_b = b.copy()

    slack_indices = list(range(num_vars, num_vars + num_ineq_constraints))

    return new_Q, new_p, new_G, new_H, new_A, new_b,slack_indices


def remove_slack_variables(solution, slack_indices):
    solution = np.array(solution)

    filtered_solution = np.delete(solution, slack_indices)

    return filtered_solution


def solve_and_store(Q, p, G, H, A, b):
    solution = solve_qp(Q, p, G, H, A, b, solver='cvxopt')
    if solution is not None:
        return solution

    Q = make_positive_semidefinite(Q)
    Q, p, G, H, A, b,slack_indices =add_slack_variables(Q, p, G, H, A, b)
    solution = solve_qp(Q, p, G, H, A, b, solver='cvxopt')
    solution = remove_slack_variables(solution, slack_indices)

    if solution is None:
        print('can not get feasible solution')
    return solution

