import numpy as np
from qpsolvers import solve_qp


def make_positive_semidefinite(Q, epsilon=1e-6):
    """
    将矩阵 Q 调整为半正定矩阵，通过添加 ε*I 来确保它的正定性。

    参数:
    Q (numpy.ndarray): 原始的二次项矩阵
    epsilon (float): 调整的正定性系数

    返回:
    Q_new (numpy.ndarray): 调整后的半正定矩阵
    """
    # 检查 Q 是否已经是半正定
    eigenvalues = np.linalg.eigvals(Q)
    if np.all(eigenvalues >= 0):
        print("Q is already positive semidefinite.")
        return Q

    # 添加 ε*I 来确保正定性
    Q_new = Q + epsilon * np.eye(Q.shape[0])
    return Q_new
def add_slack_variables(Q, p, G, H, A, b, slack_weight=10):
    """
    为不等式约束 (G, H) 添加松弛变量，并返回修改后的 Q, p, G, H, A, b。

    参数:
        Q: 原目标函数中的二次项矩阵 (n x n)
        p: 原目标函数中的线性项向量 (n,)
        G: 原不等式约束矩阵 (m x n)
        H: 原不等式约束右端项向量 (m,)
        A: 原等式约束矩阵 (k x n)
        b: 原等式约束右端项向量 (k,)
        slack_weight: 松弛变量的惩罚权重，默认为 10

    返回:
        修改后的 Q, p, G, H, A, b，其中松弛变量被添加到 G 和 H 中。
    """
    num_vars = Q.shape[0]  # 原始变量数量
    num_ineq_constraints = G.shape[0]  # 不等式约束数量
    num_eq_constraints = A.shape[0]  # 等式约束数量

    # 扩展 Q 矩阵和 p 向量
    new_Q = np.pad(Q, ((0, num_ineq_constraints), (0, num_ineq_constraints)), 'constant')
    new_p = np.pad(p, (0, num_ineq_constraints), 'constant')

    # 在 Q 中为每个松弛变量添加一个惩罚项
    for i in range(num_ineq_constraints):
        new_Q[num_vars + i, num_vars + i] = 2.0 * slack_weight  # 给松弛变量添加权重

    # 扩展 G 矩阵
    new_G = np.pad(G, ((0, 0), (0, num_ineq_constraints)), 'constant')

    # 为每个不等式约束添加对应的松弛变量
    for i in range(num_ineq_constraints):
        new_G[i, num_vars + i] = 1.0  # 松弛变量的系数是 1

    # H 不变，因为松弛变量的值会被解决，不需要调整 H
    new_H = H.copy()

    # 扩展 A 矩阵，为每个松弛变量添加零列
    new_A = np.pad(A, ((0, 0), (0, num_ineq_constraints)), 'constant')

    # b 不变，因为松弛变量不影响等式约束
    new_b = b.copy()

    # 返回松弛变量的位置，它们在新的 Q 矩阵中的索引为 [num_vars, num_vars + 1, ..., num_vars + num_ineq_constraints - 1]
    slack_indices = list(range(num_vars, num_vars + num_ineq_constraints))

    return new_Q, new_p, new_G, new_H, new_A, new_b,slack_indices


def remove_slack_variables(solution, slack_indices):
    """
    从包含松弛变量的解中移除松弛变量，返回仅包含原始变量的解。

    参数:
        solution: 包含松弛变量的解 (n + m,)
        slack_indices: 松弛变量的索引列表

    返回:
        不包含松弛变量的原始解 (n,)
    """
    # 将解转换为列表或数组，确保我们可以删除元素
    solution = np.array(solution)

    # 删除松弛变量对应的元素
    filtered_solution = np.delete(solution, slack_indices)

    return filtered_solution


# 示例用法：
# 假设 Q, p, G, H, A, b 已经定义为原始问题的参数
# Q, p, G, H, A, b = add_slack_variables(Q, p, G, H, A, b, slack_weight=10)

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

