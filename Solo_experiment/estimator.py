import numpy as np
import crocoddyl
import pinocchio as pin
from model import ActionModelCOM
import matplotlib.pyplot as plt

import os
import sys
src_path = os.path.abspath("../")
sys.path.append(src_path)
from filters.EKF import EKF
from filters.ERSKF import ERSKF
from model import measurement_model, ActionModelCOM_EST



class RobotCenteroidalEstimator():

    def __init__(self, robot, n_col, eff_names, dt = 0.001):
        """
        Input:
            robot : robot object from robot wrapper
            n_col : number of collocation points
            eff_names : names of end effectors
            dt : discretization
        """
        self.rmodel = robot.model
        self.rdata = robot.data
        self.dt = dt
        self.m = pin.computeTotalMass(self.rmodel)

        self.n_eff = len(eff_names)
        self.eff_names = eff_names
        self.ee_frame_id = []
        for i in range(len(self.eff_names)):
            self.ee_frame_id.append(self.rmodel.getFrameId(self.eff_names[i]))        
        
    def init_filter(self, q0, v0, P0, Q, R, Df, sigma, estimator):
        mes_model = measurement_model(15, 9)
        running_est = ActionModelCOM_EST(self.r, self.m, self.xref, self.xcoeff, self.ucoeff, dt=self.dt)
        x0 = self.get_centroidal_state(q0, v0)
        x0_hat = np.concatenate([x0, Df])
        self.estimator = estimator
        if self.estimator == "EKF" :
            self.ekf = EKF(x0_hat, P0, Q, R, running_est, mes_model)
        elif self.estimator == "ERSKF":
            self.ekf = ERSKF(x0_hat, P0, Q, R, sigma, running_est, mes_model)
        elif self.estimator == "NA":
            pass 
        
        return x0_hat 

    def set_cost_params(self, r, xref, xcoeff, ucoeff):
        """
        Sets the parameters of the optimization problem (costs)
        r (list): Contact locations
        xref (list): desired state (x, v, L)
        xcoeff (list): tracking coefficient
        ucoeff (list): control coefficient
        """
        self.xcoeff = xcoeff
        self.ucoeff = ucoeff
        self.xref = xref
        self.r = r

    def compute_ori_correction(self, q, des_quat):
        """
        This function computes the AMOM required to correct for orientation
        q : current joint configuration
        des_quat : desired orientation
        """
        pin_quat = pin.Quaternion(np.array(q[3:7]))
        pin_des_quat = pin.Quaternion(np.array(des_quat))

        omega = pin.log3((pin_des_quat*(pin_quat.inverse())).toRotationMatrix())

        return omega

    def compute_contact_location(self):
        """
        This function returns the location of the end effectors
        """
        self.r = np.zeros((self.n_eff, 3))
        for i in range(self.n_eff):
            self.r[i] = np.round(self.rdata.oMf[self.ee_frame_id[i]].translation, 3) 
        return self.r

    def update_action_model(self, xref, r):
        """
        Sets the parameters of the optimization problem (costs)
        xref (list): desired state (x, v, L)
        r (list): Contact locations
        """
        if self.estimator != "NA":
            self.ekf.process_model.update_params(xref, r) 

    def get_centroidal_state(self, q, v):        
        """
        Input:
            q : current joint configuration of robot
            v : current velocities of joints
        """
        # updating model and data
        pin.forwardKinematics(self.rmodel, self.rdata, q, v)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        pin.computeCentroidalMomentum(self.rmodel, self.rdata)
        # defining x
        x = np.zeros(9)
        x[0:3] = pin.centerOfMass(self.rmodel, self.rdata, q.copy(), v.copy())
        x[3:] = np.array(self.rdata.hg)
        x[3:6] /= self.m
        return x

    def step(self, q, v, u_prev, x_ref, r, Vxx=None, Vx=None):
        """
        Input:
            q : current joint configuration of robot
            v : current velocities of joints
            u_prev : last control input
            Vxx : Value function hessian at the next time step
            Vx : Value function gradient at the next time step
        """
        self.update_action_model(x_ref, r)

        y = self.get_centroidal_state(q, v)
        if self.estimator == "EKF":
            x, _ = self.ekf.step(u_prev, y)
        elif self.estimator == "ERSKF":
            x, _ = self.ekf.step(u_prev, y, Vxx, Vx)
        elif self.estimator == "NA":
            x = np.zeros(15)
            x[0:9] =  self.get_centroidal_state(q, v)
            
        return x

    def step_parallel(self, channel):
        while True:
            q, v, u_prev, Vxx, Vx, x_ref, r = channel.recv()
            x = self.step(q, v, u_prev, x_ref, r, Vxx, Vx)
            channel.send((x))
