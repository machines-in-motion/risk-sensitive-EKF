## This is the DG_head implementation of the ERSKF EKF based controller 


import numpy as np
import pinocchio as pin
from multiprocessing import Process, Pipe
from planner import RobotCenteroidalPlanner
from estimator import RobotCenteroidalEstimator
from blmc_controllers.robot_impedance_controller import RobotImpedanceController

import time
import pybullet as p

class DGHeadERSKFController:

    def __init__(self, head, vicon_name, config_file, pin_robot, f_arr, n_col, run_sim=False):

        self.t = 0
        self.pin_robot = pin_robot
        self.rmodel = self.pin_robot.model
        self.rdata = self.pin_robot.data
        self.f_arr = f_arr
        self.n_col = n_col
        self.vicon_name = vicon_name
        self.run_sim = run_sim

        # Controller
        # Impedance controller gains
        self.kp = len(f_arr) * [0, 0, 0]
        self.kd = len(f_arr) * [0, 0, 0]

        # Desired leg length.
        self.x_des = len(f_arr) * [0.0, 0.0, -0.22]
        self.xd_des = len(f_arr) * [0.0, 0.0, 0.0]

        # distributing forces to the active end-effectors
        f = np.zeros(len(f_arr) * 3)
        f = len(f_arr) * [0.0, 0.0, 0]

        self.robot_leg_ctrl = RobotImpedanceController(pin_robot, config_file.ctrl_path)

    

        self.q0 = np.array(config_file.initial_configuration)
        self.v0 = np.array(config_file.initial_velocity)

        ## setting up pipes

        self.estimator_parent, self.estimator_child = Pipe()
        self.planner_parent, self.planner_child = Pipe()

        n_filt = 10
        self.base_pos_filter = np.zeros((3, n_filt))
        self.base_vel_filter = np.zeros((3, n_filt))


        ## getting joint positions
        self.head = head
        self.joint_positions = self.head.get_sensor('joint_positions')
        self.joint_velocities = self.head.get_sensor('joint_velocities')
        self.imu_gyroscope = head.get_sensor('imu_gyroscope')

    def init_filter(self, filter_dt, P0, Q, R, sigma, filter_type):
        self.estimator = RobotCenteroidalEstimator(self.pin_robot, self.n_col, self.f_arr, dt=filter_dt)
        self.P0 = P0
        self.Q = Q
        self.R = R
        self.sigma = sigma
        self.filter_type = filter_type

    def init_planner(self, planner_dt, xcoeff, ucoeff, des_com, des_vcom, correction_ratio=0.2):
        self.planner = RobotCenteroidalPlanner(self.pin_robot, self.n_col, planner_dt, correction_ratio)
        self.xcoeff = xcoeff 
        self.ucoeff = ucoeff
        self.des_com = des_com
        self.des_vcom = des_vcom

    def set_replan_times(self, replan_time, filter_replan_time):

        self.replan_time = replan_time
        self.filter_replan_time = filter_replan_time

    def get_base(self, thread):
        base_pos, base_vel = thread.vicon.get_state(self.vicon_name)
        self.base_pos_filter[:, 0:-1] = self.base_pos_filter[:, 1:]
        self.base_pos_filter[:, -1] = base_vel[:3]
        base_vel[:3] = np.mean(self.base_pos_filter, axis=1)
        self.base_vel_filter[:, 0:-1] = self.base_vel_filter[:, 1:]
        self.base_vel_filter[:, -1] = self.imu_gyroscope
        base_vel[3:] = np.mean(self.base_vel_filter, axis=1)
        return base_pos, base_vel

    def warmup(self,thread):

        self.planner.set_action_model(self.filter_type)
        self.planner.set_weights(self.xcoeff, self.ucoeff)

        base_pos, base_vel = self.get_base(thread)

        if np.linalg.norm(self.base_vel_filter[:, -1]) == 0 and not self.run_sim:
            print("IMU not working")
            assert False

        self.q = np.hstack([base_pos, self.joint_positions])
        self.v = np.hstack([base_vel, self.joint_velocities])
        # self.q_start = self.q.copy()

        self.q_start = self.estimator.get_centroidal_state(self.q, self.v).copy()

        self.q[0:2] = self.q[0:2] - self.q_start[0:2]

        self.x_ref = self.planner.compute_xref(self.q, self.des_com, self.des_vcom, [0,0,0,1])
        self.r = self.estimator.compute_contact_location()
        self.estimator.set_cost_params(self.r, self.x_ref, self.xcoeff, self.ucoeff)

        self.x_init = self.estimator.init_filter(self.q, self.v, self.P0, self.Q, self.R, self.sigma, self.filter_type)
        planner_subp = Process(target= self.planner.optimize_parallel, args=([self.planner_child]))
        estimator_subp = Process(target= self.estimator.step_parallel, args=([self.estimator_child]))
        planner_subp.start()
        estimator_subp.start()

        ## First planner call
        self.r = self.estimator.compute_contact_location()
        self.planner_parent.send((self.x_init, self.x_ref, self.r))
        self.xs, self.us, self.Vxx, self.Vx = self.planner_parent.recv()
        self.ctr = 0
        self.t += 1
        self.planner_ready = True
        self.filter_ready = True

    def run(self, thread):
        # print(self.q[0:3], self.q[3:7], self.v[0:3], self.v[3:6])
        
        if int(self.t % (1000*self.replan_time)) == 0 and self.planner_ready:    
            self.x_ref = self.planner.compute_xref(self.q, self.des_com, self.des_vcom, [0,0,0,1])
            self.r = self.estimator.compute_contact_location()
            self.planner_parent.send((self.x_init, self.x_ref, self.r))
            self.planner_ready = False

        if self.t != 0 and self.planner_parent.poll():
            self.xs, self.us, self.Vxx, self.Vx = self.planner_parent.recv()
            self.ctr = 0        
            self.planner_ready = True

        self.f = self.us[self.ctr]
        # print("base pos", self.q[0:3])
        # print("base orientation", self.q[3:7])
        # print("joint_positions", self.joint_positions)
        # print("joint_velocities", self.joint_velocities)
        # print("x_ref", self.x_ref)
        # print("Desired force", self.f)
        # self.f = np.zeros_like(self.f)

        tau = self.robot_leg_ctrl.return_joint_torques(self.q, self.v, \
                            self.kp, self.kd, self.x_des, self.xd_des, self.f)

        # tau = - 3 * (self.joint_positions - self.q0[7:]) - 0.05 * (self.joint_velocities - self.v0[6:])
        # print(tau)

        self.head.set_control('ctrl_joint_torques', tau)

        base_pos, base_vel = self.get_base(thread)

        self.q = np.hstack([base_pos, self.joint_positions]) 
        self.q[0:2] = self.q[0:2] - self.q_start[0:2].copy() 
        self.v = np.hstack([base_vel, self.joint_velocities])

        self.x_GT = self.estimator.get_centroidal_state(self.q, self.v)  # this will give the ground truth
        
        if int(self.t % (1000*self.filter_replan_time)) == 0 and self.filter_ready:
            self.estimator_parent.send((self.q, self.v, self.f, self.Vxx, self.Vx, self.x_ref, self.r))
            self.filter_ready = False
        
        if  self.estimator_parent.poll():
            self.x_init = self.estimator_parent.recv()
            self.filter_ready = True

        self.ctr += 1
        self.t += 1