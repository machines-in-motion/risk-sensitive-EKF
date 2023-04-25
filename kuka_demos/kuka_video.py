import numpy as np 
np.random.seed(0) # for reproductibility
import matplotlib.pyplot as plt 
import os, sys, time
import crocoddyl 
import pinocchio as pin
import os
import sys
src_path = os.path.abspath("../")
sys.path.append(src_path)
from kuka_model.kuka_circle import KukaCircle, measurement_model_full, measurement_model, get_p_
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
print("[PyBullet] Created robot (id = "+str(robot_simulator.robotId)+")")

x0 = kukacircle.x0
nx = kukacircle.nx
T = kukacircle.T


# Warm start:
ddp_solver = kukacircle.ddp_solver
xs_init = [x0 for i in range(T+1)]
us_init = ddp_solver.problem.quasiStatic(xs_init[:-1])
ddp_solver.solve(xs_init, us_init, maxiter=10, isFeasible=False)
xinit = ddp_solver.xs.tolist().copy()
uinit = ddp_solver.us.tolist().copy()


# Initialize filters
x0_hat = x0
P0 = 1e-6 * np.eye(nx)
Q = 1e-1 * np.eye(nx)
R = 1e-6 * np.eye(nx)
meas_model = measurement_model(nx, nx)
kukacircle_est = KukaCircle(robot_simulator, horizon, dt=dt, dt_sim=dt_sim)
ekf = EKF(x0_hat, P0, Q, R, kukacircle_est.runningModelfilt, meas_model)
sigma = 7.5e4
erskf = ERSKF(x0_hat, P0, Q, R, sigma, kukacircle_est.runningModelfilt, meas_model)



# Force perturbation
T_sim = int(8 / dt_sim)
impact_time = 1
T_force = int(impact_time / dt_sim)
dt_force = 500
direction = np.array([-1, 0, 1])
force = 80 * direction / np.linalg.norm(direction)
linkIndex = 6
print("\nimpact_time", impact_time)
print("force", force)



#### Reset sim
robot_simulator.reset_state(q0, v0)
robot_simulator.forward_robot(q0, v0)
x = np.concatenate([q0, v0]).T



# Video
VIDEO = False

p.resetDebugVisualizerCamera(
     cameraDistance=1.3,
     cameraYaw=30,
     cameraPitch=-32,
     cameraTargetPosition=[0.25, 0, 0],
 )


circle = list(kukacircle_est.target)
color = [[0, 0, 0]] * len(kukacircle_est.target)
# p.addUserDebugPoints(circle, color)

if VIDEO:
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    video_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "./" + "RS-EKF_EKF" + ".mp4")
        


endeff_frame_id = robot_simulator.pin_robot.model.getFrameId("contact")
prev_pose = kukacircle_est.target[0]
trailDuration = 100




#### Reset sim
q = q0
robot_simulator.reset_state(q0, v0)
robot_simulator.forward_robot(q0, v0)


#### Risk sensitive simulation

xtraj = [[x0], [x0]]
xest = [[x0], [x0]]
utraj = [[], []]
cost = [[], []]

for ii in range(2):
    xs = xinit.copy()
    us = uinit.copy()

    est = x0.copy()

    for t in range(T_sim):
        print ("\r Simulation... {} / ".format(t+1) +str(T_sim), end="")

        # Solve with DDP solver
        kukacircle.update_reference(est, t)
        ddp_solver.solve(xs, us, maxiter=10, isFeasible=False)
        u = ddp_solver.us[0]

        robot_simulator.send_joint_command(u)
        # External push
        if T_force <= t  and t < T_force + dt_force :   
            position = p.getLinkState(robot_simulator.robotId, linkIndex)[0]
            p.applyExternalForce(objectUniqueId=robot_simulator.robotId, linkIndex=linkIndex, forceObj=force, posObj=position, flags=p.WORLD_FRAME)

        env.step()

        # Measure new state from simulator 
        q, v = robot_simulator.get_state()

        # Update pinocchio model
        robot_simulator.forward_robot(q, v)
        # Record data 
        x = np.concatenate([q, v]).T
        xtraj[ii].append(x)
        utraj[ii].append(u)
        y = x
        if ii == 0:
            est, _ = erskf.step(u, y, ddp_solver.Vxx[1], ddp_solver.Vx[1])
        else:
            est, _ = ekf.step(u, y)

        xest[ii].append(est)
        #Update warm start
        xs = list(ddp_solver.xs[1:]) + [ddp_solver.xs[-1]]
        xs[0] = est
        us = list(ddp_solver.us[1:]) + [ddp_solver.us[-1]] 

xtraj = np.array(xtraj)
xest = np.array(xest)
utraj = np.array(utraj)


np.save( "kuka_push_RS-EKF", xtraj[0])
np.save( "kuka_push_EKF", xtraj[1])


if VIDEO:
    p.stopStateLogging(video_id)

# kukacircle.plot_state(xtraj[1], xest[1], xtraj[0], xest[0])
# kukacircle.plot_control(utraj[1], utraj[0])

EKF_MSE, ERSKF_MSE, mse_traj, rs_mes_traj, tspan_sim = kukacircle.plot_endeff(xtraj[1], xest[1], xtraj[0], xest[0])

EKF_MSE = np.mean(EKF_MSE)
ERSKF_MSE = np.mean(ERSKF_MSE)
print("Average MSE EKF", EKF_MSE)
print("Average MSE ERSKF", ERSKF_MSE)

gain = 100 * (ERSKF_MSE - EKF_MSE) / EKF_MSE
print("Average GAIN: ", gain)


plt.plot()