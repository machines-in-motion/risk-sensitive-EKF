import numpy as np
import crocoddyl
import matplotlib.pyplot as plt
import os
import sys

src_path = os.path.abspath("../")
sys.path.append(src_path)

from quadrotor_model.quadrotor_model import (
    MassQuadrotorEnv,
    DifferentialActionModelQuadrotor,
    ActionModelQuadrotorMass,
    measurement_model,
)
from filters.EKF import EKF
from filters.ERSKF import ERSKF
from data_recorder import DataRecorder

LINE_WIDTH = 100


mass_guess = 2.0
x0 = np.array([0, 0, 0, 0, 0, 0, mass_guess])
H = 20
T_sim = 80
dt = 0.05

xdes_ref = [t / T_sim for t in range(T_sim + H + 1)]
ydes_ref = [0 for t in range(T_sim + H + 1)]
time = np.linspace(0, dt * (T_sim + 1), T_sim + 1)


runnings = []
for i in range(T_sim + H + 1):
    quadrotor_running = ActionModelQuadrotorMass(xdes_ref[i], ydes_ref[i], dt)
    runnings.append(quadrotor_running)


# Initialize filters
meas_model = measurement_model(7, 3)
x0_hat = x0
P0 = 1e-4 * np.eye(7)
Q = 1e-4 * np.eye(7)
Q[-1, -1] = 2
R = 1e-4 * np.diag([1, 1, 1])


# Reccord cost function
def evaluate_cost(model, x, u):
    data = model.createData()
    model.calc(data, x, u)
    return data.cost


# Run simulation
mass_variation = [5] * 40 + [2] * 40
env = MassQuadrotorEnv(quadrotor_running, meas_model)
filter_type = [4e-3, 0]
dr = DataRecorder(filter_type, xdes_ref, ydes_ref, dt=dt)


for i in range(2):
    if filter_type[i] == 0:
        estimator = EKF(x0_hat, P0, Q, R, quadrotor_running, meas_model)
    else:
        estimator = ERSKF(
            x0_hat, P0, Q, R, filter_type[i], quadrotor_running, meas_model
        )

    xs = [x0_hat] * (H + 1)
    us = [np.zeros(2)] * H

    s = env.init_state(x0)
    est = x0_hat

    dr.add_experiment()

    for t in range(T_sim):
        print("\r Simulation... {} / ".format(t + 1) + str(T_sim), end="")
        problem = crocoddyl.ShootingProblem(est, runnings[t : t + H], runnings[t + H])
        ddp = crocoddyl.SolverFDDP(problem)
        # ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

        converged = ddp.solve(xs, us, maxiter=1000)
        u = ddp.us[0]
        cost = evaluate_cost(runnings[t], s, u)
        s = env.step(u)

        y = env.obs()
        if filter_type[i] == 0:
            est, P = estimator.step(u, y)
        else:
            est, P = estimator.step(u, y, ddp.Vxx[1], ddp.Vx[1])

        env.change_mass(mass_variation[t])

        xs = [est] + ddp.xs.tolist()[1:]
        us = ddp.us.tolist()[1:] + [np.zeros(2)]

        dr.record_data(est, s, u, cost)


print("\n")
dr.plot()
