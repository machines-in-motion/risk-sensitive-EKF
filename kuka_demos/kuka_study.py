import numpy as np

np.random.seed(0)  # for reproductibility
import matplotlib.pyplot as plt
import os, sys, time
import crocoddyl
import pinocchio as pin
import os
import sys

src_path = os.path.abspath("../")
sys.path.append(src_path)
from kuka_model.kuka_circle import KukaCircle, measurement_model_full, measurement_model
from bullet_utils.env import BulletEnvWithGround
from robot_properties_kuka.iiwaWrapper import IiwaRobot
import pybullet as p
import time
from filters.EKF import EKF
from filters.ERSKF import ERSKF

dt_sim = 2e-3
env = BulletEnvWithGround(p.DIRECT, dt=dt_sim)
robot_simulator = IiwaRobot()
horizon = 20
dt = 0.05
kukacircle = KukaCircle(robot_simulator, horizon, dt=dt, dt_sim=dt_sim)
env.add_robot(robot_simulator)
x0 = kukacircle.x0
q0 = kukacircle.q0
v0 = kukacircle.v0
robot_simulator.reset_state(q0, v0)
robot_simulator.forward_robot(q0, v0)
print("[PyBullet] Created robot (id = " + str(robot_simulator.robotId) + ")")

x0 = kukacircle.x0
nx = kukacircle.nx
T = kukacircle.T


# Warm start : initial state + gravity compensation
ddp_solver = kukacircle.ddp_solver
xs_init = [x0 for i in range(T + 1)]
us_init = ddp_solver.problem.quasiStatic(xs_init[:-1])
ddp_solver.solve(xs_init, us_init)
xref = ddp_solver.xs
uref = ddp_solver.us


MSE_LIST = [[], []]
MSE_TRAJ_LIST = [[], []]
COST_LIST = [[], []]
GAIN_LIST = []

for expe in range(10000):
    print("EXPE INDEX ", expe)

    # Initialize filters
    x0_hat = x0
    P0 = 1e-6 * np.eye(nx)
    Q = 1e-1 * np.eye(nx)
    R = 1e-6 * np.eye(nx)
    meas_model = measurement_model(nx, nx)
    kukacircle_est = KukaCircle(robot_simulator, horizon, dt=dt, dt_sim=dt_sim)
    ekf = EKF(x0_hat, P0, Q, R, kukacircle_est.runningModelfilt, meas_model)

    sigma = 7.5e4
    erskf = ERSKF(
        x0_hat, P0, Q, R, sigma, kukacircle_est.runningModelfilt, meas_model
    )

    # Force perturbation
    T_sim = int(4 / dt_sim)
    # impact_time = 1.
    impact_time =  1 + np.random.random() * 2

    T_force = int(impact_time / dt_sim)
    dt_force = 500

    # direction = np.array([-1, 0, 1])
    direction = 2 * np.random.random(3) - 1
    force = 80 * direction / np.linalg.norm(direction)
    linkIndex = 6

    print("\nimpact_time", impact_time)
    print("force", force)


    xtraj = [[x0], [x0]]
    xest = [[x0], [x0]]
    utraj = [[], []]
    cost = [[], []]

    for ii in range(2):
        #### Reset sim
        robot_simulator.reset_state(q0, v0)
        robot_simulator.forward_robot(q0, v0)
        x = np.concatenate([q0, v0]).T

        #### Risk sensitive simulation
        est = x0

        xs = xref
        us = uref


        for t in range(T_sim):
            # print("\r RS Simulation... {} / ".format(t + 1) + str(T_sim), end="")

            # Solve with DDP solver
            kukacircle.update_reference(est, t)
            # ddp_solver = kukacircle.create_solver(est, t)
            ddp_solver.solve(xs, us, maxiter=1000, isFeasible=False)   
            # ddp_solver.solve()   


            u = ddp_solver.us[0]

            ddp_solver.problem.runningModels[0].calc(
                ddp_solver.problem.runningDatas[0], x, u
            )
            cost[ii].append(ddp_solver.problem.runningDatas[0].cost)

            robot_simulator.send_joint_command(u)

            # External push
            if T_force <= t and t < T_force + dt_force:
                position = p.getLinkState(robot_simulator.robotId, linkIndex)[0]
                p.applyExternalForce(
                    objectUniqueId=robot_simulator.robotId,
                    linkIndex=linkIndex,
                    forceObj=force,
                    posObj=position,
                    flags=p.WORLD_FRAME,
                )

            env.step()

            # Measure new state from simulator
            q, v = robot_simulator.get_state()
            # Update pinocchio model
            robot_simulator.forward_robot(q, v)
            # Record data
            x = np.concatenate([q, v]).T
            xtraj[ii].append(x)
            utraj[ii].append(u)
            # print(xtraj[ii])
            y = x
            if ii == 0:
                est, _ = erskf.step(u, y, ddp_solver.Vxx[1], ddp_solver.Vx[1])
            else:
                est, _ = ekf.step(u, y)

            xest[ii].append(est)

            # Update warm start
            xs = list(ddp_solver.xs[1:]) + [ddp_solver.xs[-1]]
            xs[0] = est
            us = list(ddp_solver.us[1:]) + [ddp_solver.us[-1]]

        COST_LIST[ii].append(cost[ii])


    xtraj = np.array(xtraj)
    xest = np.array(xest)
    utraj = np.array(utraj)


    # kukacircle.plot_state(xtraj[1], xest[1], xtraj[0], xest[0])
    # kukacircle.plot_control(utraj[1], utraj[0])
    EKF_MSE, ERSKF_MSE, mse_traj, rs_mes_traj, tspan_sim = kukacircle.plot_endeff(xtraj[1], xest[1], xtraj[0], xest[0])

    MSE_LIST[0].append(ERSKF_MSE)
    MSE_LIST[1].append(EKF_MSE)

    MSE_TRAJ_LIST[0].append(rs_mes_traj)
    MSE_TRAJ_LIST[1].append(mse_traj)

    print("\n")
    print("MSE EKF", EKF_MSE)
    print("MSE ERSKF", ERSKF_MSE)

    gain = 100 * (ERSKF_MSE - EKF_MSE) / EKF_MSE
    print("GAIN: ", gain)
    GAIN_LIST.append(gain)

    plt.show()

plt.close("all")

EKF_MSE_TRAJ_LIST = np.array(MSE_TRAJ_LIST[1])
ERSKF_MSE_TRAJ_LIST = np.array(MSE_TRAJ_LIST[0])


EKF_COST = np.array(COST_LIST[1])
ERSKF_COST = np.array(COST_LIST[0])

np.save("ekf_cost", EKF_COST)
np.save("rs-cost", ERSKF_COST)

np.save("ekf_mse", EKF_MSE_TRAJ_LIST)
np.save("rs-ekf_mse", ERSKF_MSE_TRAJ_LIST)


# import pdb; pdb.set_trace()
EKF_MSE_MEAN_LIST = np.mean(EKF_MSE_TRAJ_LIST, axis=0)
ERSKF_MSE_MEAN_LIST = np.mean(ERSKF_MSE_TRAJ_LIST, axis=0)

EKF_MSE_STD_LIST = np.std(EKF_MSE_TRAJ_LIST, axis=0)
ERSKF_MSE_STD_LIST = np.std(ERSKF_MSE_TRAJ_LIST, axis=0)

print("------------Computing average--------------")

EKF_MSE = np.mean(MSE_LIST[1])
ERSKF_MSE = np.mean(MSE_LIST[0])
print("Average MSE EKF", EKF_MSE)
print("Average MSE ERSKF", ERSKF_MSE)

gain = 100 * (ERSKF_MSE - EKF_MSE) / EKF_MSE
print("Average GAIN: ", gain)
print("Average GAIN 2: ", np.mean(GAIN_LIST))
