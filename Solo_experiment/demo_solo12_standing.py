## This a demo of solo12 standing

import os
import time
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Process, Pipe

from robot_properties_solo.solo12wrapper import Solo12Config
from planner import RobotCenteroidalPlanner
from estimator import RobotCenteroidalEstimator
from data_recorder import DataRecorder
from utils import Inverse_Dynamics

# os.nice(-20)

SIM = False
RECORD = False
filter_type = "ERSKF" # "ERSKF", "EKF"
# filter_type = "ERSKF" # "ERSKF", "EKF"
EXPERIMENT_DURATION = 30 #s

# dz_init = +20
# extra_logname = f"_dz_{dz_init}_CUBE_FILMED"
# Df_init = np.array([0, 0, dz_init, 0, 0, 0])


extra_logname = f"_dropped_FILMED"
Df_init = np.array([0, 0, 0, 0, 0, 0])


## robot config andpost
pin_robot = Solo12Config.buildRobotWrapper()
f_arr = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]


# init robot
dt_ctrl = 1 * 1e-3
if SIM:
    dt_ctrl = 1 * 1e-3

robot_config = Solo12Config()
q0 = np.array(robot_config.initial_configuration)
# q0 = np.array([0.2, 0.0, 0.12, 0.0, 0.0, 0.0, 1.0, 0.0, 1.2, -2.4, 0.0, 1.2, -2.4, 0.0, -1.2, 2.4, 0.0, -1.2, 2.4])
v0 = np.array(robot_config.initial_velocity)


## init planner
n_col = 4
dt_plan = 0.05
planner_parent, planner_child = Pipe()
planner_ready = True
planner = RobotCenteroidalPlanner(pin_robot, n_col, dt_ctrl, dt_plan)
xcoeff = [1e2, 1e2, 1e2, 1e1, 1e1, 1e1, 1e1, 1e1, 1e1]
ucoeff = [1e-4, 1e-4, 1e-6] * len(f_arr) * 3


if(SIM):
    from bullet_utils.env import BulletEnvWithGround
    from robot_properties_solo.solo12wrapper import Solo12Robot
    from utils import Disturbances, get_contact, Block
    import pybullet as p

    env = BulletEnvWithGround(p.GUI, dt=dt_ctrl)
    robot = Solo12Robot()
    robot = env.add_robot(robot)
    def get_robot_state():
        return robot.get_state()
    def get_pos_ref():
        return None
else:
    import threading
    # Create robot interface (if)
    import libodri_control_interface_pywrap as oci
    robot_if = oci.robot_from_yaml_file("/home/solo/devel/workspace/src/odri_control_interface/demos/config_solo12.yaml")

    from qualisys2vicon_sdk import ViconClient, ViconFrame
    vicon_client = ViconClient()
    vicon_client.initialize("192.168.75.120")
    vicon_client.run()
    pos_ref = None
    def get_robot_state():
        # Intrisics measures
        robot_if.parse_sensor_data()
        w_imu = robot_if.imu.gyroscope
        q_i = robot_if.joints.positions
        v_i = robot_if.joints.velocities

        # Extrinsics measures
        frame = vicon_client.get_vicon_frame()
        pos = frame.se3_pose[:3]
        quat = frame.se3_pose[3:7]
        v_e = frame.velocity_world_frame[:3]

        global pos_ref
        if(pos_ref is None):
            pos_ref = pos[:]

        pos_relat = pos - pos_ref

        q = np.concatenate([pos_relat, quat, q_i])
        v = np.concatenate([v_e, w_imu, v_i])

        return q, v
    def get_pos_ref():
        global pos_ref
        return pos_ref


ID = Inverse_Dynamics(pin_robot, dt_ctrl)


# desired centor of mass
# des_com = [0, 0, 0.08]
des_com = [0, 0, 0.0]
des_vcom = [0,0,0]
if(SIM):
    # Compensating for PyBullet not spawning the robot at the origin
    des_com = [0.2, 0, 0.2]
    # des_com = [0.2, 0, 0.1]

# EKF filter
estimator_parent, estimator_child = Pipe()
filter_dt = 0.005
filter_ready = True
estimator = RobotCenteroidalEstimator(pin_robot, n_col, f_arr, dt=filter_dt)
x_ref = planner.compute_xref(q0, des_com, des_vcom, [0,0,0,1])
r = estimator.compute_contact_location()
estimator.set_cost_params(r, x_ref, xcoeff, ucoeff)

# P0 = np.diag([1e-4]*9 +  [1e-2]*3 + [1e-2]*3)
Q = np.diag([1e-3]*6 + [1e-4]*3 + [1e-1]*3 + [1e-2]*3)
P0 = Q
R = np.diag([1e-4]*3 + [1e-2]*3 + [1e-4]*3 )
sigma = 6 # 6 or 7 is limit

# Simulation parameters
T_experiment = int(EXPERIMENT_DURATION / dt_ctrl)
if(SIM):
    t1 = int(4 / dt_ctrl)
    t2 = int(6 / dt_ctrl)
    time_seq = [t1, t2]
    torques = [[0, 0, 1], [0, 0, 0]]
    forces = [[0, 0, 0], [0, 0, 0]]
    # disturbances = Disturbances(robot.robotId, estimator, T_experiment, time_seq, forces, torques)
    disturbances = Block(robot.robotId, filter_type, RECORD)

# MPC freq
replan_time = 0.01
filter_replan_time = filter_dt
ctr = 0


# data recorder
dr = DataRecorder()
dr.estimator_type = filter_type
dr.dt = dt_ctrl
dr.add_experiment(T_experiment)


# Run realtime
pln_realtime = True
est_realtime = True


## Updating planner info based on filter type
planner.set_action_model(filter_type)
planner.set_weights(xcoeff, ucoeff)
planner_subp = Process(target= planner.optimize_parallel, args=([planner_child]))
planner_subp.start()


n_filt = 8
v_table = np.zeros((18, n_filt))


# Run experiment
if(SIM):
    robot.reset_state(q0, v0)
else:
    robot_if.initialize(q0[7:])
    def get_input():
        keystrk = input()
    th = threading.Thread(target=get_input)
    th.start()
    Kp_pos, Kd_pos = 6.0, 0.3
    robot_if.joints.set_torques(np.zeros(12))
    robot_if.joints.set_desired_positions(q0[7:])
    robot_if.joints.set_desired_velocities(np.zeros(12))
    robot_if.joints.set_position_gains(Kp_pos*np.ones(12))
    robot_if.joints.set_velocity_gains(Kd_pos*np.ones(12))
    print("Put robot on the ground and press enter")
    while not robot_if.is_timeout and not robot_if.has_error and th.is_alive():
        robot_if.parse_sensor_data()
        robot_if.send_command_and_wait_end_of_cycle(dt_ctrl)



q, v = get_robot_state() # No need to change between SIM and REAL as the function has been overloaded
x_init = estimator.init_filter(q, v, P0, Q, R, Df_init, sigma, filter_type)
estimator_subp = Process(target= estimator.step_parallel, args=([estimator_child]))
estimator_subp.start()


dr.pose_ref = get_pos_ref()
tmp = []

try:
    print("Running Loop !")
    x_COM_mes = estimator.get_centroidal_state(q, v)
    for t in range(T_experiment):
        if((10)*t % T_experiment == 0):
            print("Progress: ", (10)*t/T_experiment, "/10")
        if(SIM):
            Delta_force = disturbances.applyExternalDisturbance(t, q, v)

        t1 = time.time()
        if int(t % (1000*replan_time)) == 0 and planner_ready:
            x_ref = planner.compute_xref(q, des_com, des_vcom, [0,0,0,1])
            r = estimator.compute_contact_location()
            planner_parent.send((t, x_init, x_ref, r))
            if not pln_realtime:
                t0, xs, us, Vxx, Vx, K= planner_parent.recv()
                ctr = t - t0
            else:
                planner_ready = False


        if t == 0 and pln_realtime:
            t0, xs, us, Vxx, Vx, K = planner_parent.recv()
            planner_ready = True
            ctr = t - t0

        elif t != 0 and planner_parent.poll() and pln_realtime:
            t0, xs, us, Vxx, Vx, K = planner_parent.recv()
            ctr = t - t0
            planner_ready = True
        # t2 = time.time()

        f = us[ctr] - K[ctr][:, :9] @ (x_init[:9] - xs[ctr][:9])

        ddq , f, tau = ID.get_torque(q, v, f, SIM)
        v_des = v + dt_ctrl * ddq

        if(SIM):
            robot.send_joint_command(tau - 0.06 * np.sign(v[6:]))
            env.step()
            # time.sleep(0.001)
        else:
            assert robot_if.is_ready and not robot_if.is_timeout and not robot_if.has_error, f"Error with robot: (ready - {robot_if.is_ready}) (timeout - {robot_if.is_timeout}) (has_error - {robot_if.has_error})"
            # Kp, Kd = 6.0, 0.3
            # robot_if.joints.set_torques(np.zeros(12))
            Kp, Kd = 0, 0
            robot_if.joints.set_torques(tau)

            robot_if.joints.set_desired_positions(q0[7:])
            robot_if.joints.set_desired_velocities(np.zeros(12))
            robot_if.joints.set_position_gains(Kp * np.ones(12))
            robot_if.joints.set_velocity_gains(Kd * np.ones(12))
            robot_if.send_command_and_wait_end_of_cycle(dt_ctrl)


        q, v = get_robot_state()
        v_table[:, 0:-1] = v_table[:, 1:]
        v_table[:, -1] = v
        v = np.mean(v_table, axis=1)


        x_COM_mes = estimator.get_centroidal_state(q, v)  # this will give the ground truth

        if(t%100 == 0):
            print(x_init[11])

        if int(t % (1000*filter_replan_time)) == 0 and filter_ready:
            estimator_parent.send((q, v, f, Vxx, Vx, x_ref, r))
            if not est_realtime:
                x_init = estimator_parent.recv()
            else:
                filter_ready = False

        if t == 0 and est_realtime:
            x_init = estimator_parent.recv()
            filter_ready = True

        elif t != 0 and estimator_parent.poll() and est_realtime:
            x_init = estimator_parent.recv()
            filter_ready = True


        if(SIM):
            f_GT = get_contact(robot.robotId, robot.bullet_endeff_ids)
            dr.record_data(t, x_init, x_COM_mes, f, q, v, v_des, tau, Delta_force, f_GT)
        else:
            dr.record_data(t, x_init, x_COM_mes, f, q, v, v_des, tau)

        t2 = time.time()
        ctr += 1
        tmp.append(1000*(t2 - t1))

    if RECORD:
        p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4, filter_type + ".mp4")

except Exception as e:
    print("")
    print("Error occurred while running loop")
    print(e)
    print("")
finally:
    print("Done")
    if(not SIM):
        Kp_end, Kd_end = 6.0, 0.3
        T_end = 3. #s
        q_end = get_robot_state()[0][7:] #Keep only the joint position
        N_it = int(T_end/dt_ctrl)
        for i in range(N_it):
            coeff = 1.0 - i / N_it
            robot_if.joints.set_torques(np.zeros(12))
            robot_if.joints.set_desired_positions(q_end)
            robot_if.joints.set_desired_velocities(np.zeros(12))
            robot_if.joints.set_position_gains(coeff * Kp_end * np.ones(12))
            robot_if.joints.set_velocity_gains(coeff * Kd_end * np.ones(12))
            robot_if.send_command_and_wait_end_of_cycle(dt_ctrl)

    plt.plot(tmp)
    # plt.show()
    print(np.mean(tmp))
    planner_subp.terminate()
    estimator_subp.terminate()


import os, itertools
# log function
def unique_file(basename):
    actualname = "%s" % (basename)
    c = itertools.count()
    while os.path.exists("data/" + actualname):
        actualname = "%s_%d" % (basename, next(c))
    return actualname

basename = filter_type + extra_logname
log_name = unique_file(basename)

dr.save(log_name)
dr.plot()

