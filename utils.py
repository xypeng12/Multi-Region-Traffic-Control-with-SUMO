import numpy as np
from qpsolvers import solve_qp
from scipy.optimize import linprog
import itertools

import numpy as np
import itertools
from scipy.optimize import linprog

import numpy as np
from scipy.optimize import minimize

import numpy as np
import itertools
from scipy.optimize import minimize

def minimize_with_relaxation(objective, x0, constraints, bounds,
                             slack_weight=100.0,
                             method='SLSQP',
                             verbose=True):
    """
    给所有约束都加松弛变量 (一次性添加)，并在目标函数中用 slack_weight*sum(s_i) 做惩罚。

    - ineq: con['fun'](x) >= 0  => con['fun'](x) + s_i >= 0
    - eq:   con['fun'](x) = 0   =>
           con['fun'](x) + s_i >= 0
          -con['fun'](x) + s_i >= 0
      （|fun(x)| <= s_i）
    - s_i >= 0

    返回：如果成功则返回去掉 slack 的解 x; 否则 None
    """
    n = len(x0)

    # 分析多少个约束要加 slack
    # - 对 ineq: 1 个 slack
    # - 对 eq:   1 个 slack，但会生成 2 条 ineq
    slack_defs = []
    for i, con in enumerate(constraints):
        ctype = con['type']
        if ctype == 'ineq':
            slack_defs.append(i)  # 这个约束加一个 slack
        elif ctype == 'eq':
            slack_defs.append(i)  # eq 也加一个 slack
        else:
            raise ValueError(f"Unknown constraint type {ctype}")

    # slack_count = len(slack_defs)
    # x_full = [x_1, x_2, ..., x_n, s_1, ..., s_slack_count]
    # slack 对应 slack_defs 的顺序
    slack_count = len(slack_defs)
    x0_full = np.concatenate([x0, np.ones(slack_count) * 1.0])

    # 扩展 bounds
    bounds_full = list(bounds) + [(0, None)] * slack_count  # s_i >= 0

    # 重新构造新的约束
    new_constraints = []

    # 给定 idx => slack_idx（在 x_full 里的位置）
    def make_ineq_fun(con_fun, s_idx):
        # 原: con_fun(x) >= 0 => con_fun(x) + s >= 0
        def f(x_full):
            x_main = x_full[:n]
            s_val = x_full[n + s_idx]
            return con_fun(x_main) + s_val

        return f

    def make_eq_funs(con_fun, s_idx):
        # 原: con_fun(x) = 0 =>
        #     con_fun(x) + s >= 0
        #    -con_fun(x) + s >= 0
        def f1(x_full):
            x_main = x_full[:n]
            s_val = x_full[n + s_idx]
            return con_fun(x_main) + s_val

        def f2(x_full):
            x_main = x_full[:n]
            s_val = x_full[n + s_idx]
            return -con_fun(x_main) + s_val

        return f1, f2

    # 遍历 constraints, 如果是 ineq, 做 1 条；如果 eq, 做 2 条
    s_index = 0
    for i, con in enumerate(constraints):
        ctype = con['type']
        cfun = con['fun']

        if i not in slack_defs:
            # 理论上不会发生，因为我们对每个约束都加 slack
            # 但如果你想区分“有的约束不允许放松”，可以这里处理
            new_constraints.append(con)
        else:
            # 这个约束要加 slack
            if ctype == 'ineq':
                new_constraints.append({
                    'type': 'ineq',
                    'fun': make_ineq_fun(cfun, s_index)
                })
                s_index += 1
            elif ctype == 'eq':
                f1, f2 = make_eq_funs(cfun, s_index)
                new_constraints.append({'type': 'ineq', 'fun': f1})
                new_constraints.append({'type': 'ineq', 'fun': f2})
                s_index += 1

    # 包装目标函数
    def wrapped_objective(x_full):
        x_main = x_full[:n]
        s_part = x_full[n:]
        return objective(x_main) + slack_weight * np.sum(s_part)

    # 调用 minimize
    result = minimize(
        fun=wrapped_objective,
        x0=x0_full,
        bounds=bounds_full,
        constraints=new_constraints,
        method=method,
        options={'maxiter': 1000,'ftol': 1e-7,'eps':1e-8,'disp':True}
    )

    # 无论成功与否，都返回 result.x[:n].
    if verbose:
        if result.success:
            print("✅ Minimization success with slacks.")
        else:
            print("❌ Minimization failed:", result.message)
            print("   Still returning the best solution found so far...")

    return result.x[:n]


def check_feasibility(G, H, p, bounds):
    """
    检查删除某些约束后，线性规划是否可行
    :param G: 约束系数矩阵
    :param H: 约束右端项
    :param p: 目标函数系数
    :param bounds: 变量范围
    :return: (是否可行, 线性规划解)
    """
    solution = linprog(p, A_ub=G, b_ub=H, method='highs', bounds=bounds)
    return solution.success, solution  # 返回是否可行和解


def solve_with_relaxation(n, p, G, H, bounds):
    """
    逐步删除导致无解的约束并重新求解（从删除 1 条到删除多条）
    但不删除前 n 个约束，只考虑删除第 n+1, n+2, ... 条约束。

    :param n: region number（前 n 个约束不能删除）
    :param p: 目标函数系数
    :param G: 约束系数矩阵
    :param H: 约束右端项
    :param bounds: 变量范围
    :return: 线性规划解或 None（如果仍无解）
    """
    G = np.array(G, dtype=float)
    H = np.array(H, dtype=float)
    feasible, solution=check_feasibility(G, H, p, bounds)
    if feasible:
        return solution  # 找到可行解，直接返回

    start_index = n  # 只允许删除 n+1 及之后的约束
    num_constraints = len(H)

    # 逐步尝试删除 1 条、2 条、3 条......直到找到可行解
    for num_remove in range(1, num_constraints - start_index + 1):
        for indices in itertools.combinations(range(start_index, num_constraints), num_remove):
            # 生成删除 num_remove 条约束的所有组合
            G_new = np.delete(G, indices, axis=0)  # 删除选中的约束
            H_new = np.delete(H, indices, axis=0)

            feasible, solution = check_feasibility(G_new, H_new, p, bounds)
            if feasible:
                print(f"Removing constraints {indices} made the problem feasible.")
                return solution  # 找到可行解，立即返回

    print("Error: No feasible solution found after constraint relaxation.")
    return None  # 仍然无解，返回 None

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

