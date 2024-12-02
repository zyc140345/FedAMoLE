import cvxpy as cp
import numpy as np


def optimize_expert_dispatch(prob_gate: np.ndarray, args) -> np.ndarray:
    # prob_gate: (num_experts, num_clients)

    # Parameters
    expert_choices = args.expert_choices
    max_experts = args.max_experts
    num_experts, num_clients = prob_gate.shape

    expert_dispatch = cp.Variable((num_experts, num_clients), boolean=True)  # Variables
    objective = cp.Maximize(cp.sum(cp.multiply(prob_gate, expert_dispatch)))
    constraints = [
        cp.sum(expert_dispatch, axis=1) == expert_choices,
        cp.sum(expert_dispatch, axis=0) >= 2,
        cp.sum(expert_dispatch, axis=0) <= max_experts
    ]

    # Optimize problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCIP)

    # Check if the model was solved to optimality
    if problem.status == cp.OPTIMAL:
        return expert_dispatch.value
    else:
        raise Exception(f"Optimization was not successful. Status code: {problem.status}")
