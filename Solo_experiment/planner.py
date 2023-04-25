## This class creates the DDP problem to generate trajectories for solo

import time 
import numpy as np
import crocoddyl
import pinocchio as pin
from model import ActionModelCOM, ActionModelCOM_EST
import matplotlib.pyplot as plt

LINE_WIDTH = 100

class RobotCenteroidalPlanner():

    def __init__(self, robot, n_col, dt_ctrl, dt = 0.05, correction_ratio=0.2):
        """
        Input:
            robot : robot object from robot wrapper
            n_col : number of collocation points
            dt : discretization
        """
        self.dt_ctrl = dt_ctrl
        self.rmodel = robot.model
        self.rdata = robot.data
        self.dt = dt
        self.n_col = n_col 
        self.correction_ratio = correction_ratio
        
        self.m = pin.computeTotalMass(self.rmodel)

        self.ncontact = 4
               
    def set_action_model(self, estimator):
        self.estimator = estimator

        #creating problem        
        if self.estimator == "EKF" or self.estimator == "NA":
            self.running = ActionModelCOM([None]*self.ncontact, self.m, None, None, None)
            self.terminal = ActionModelCOM([None]*self.ncontact, self.m, None, None,None, isTerminal=True)

        if self.estimator == "ERSKF":
            self.running = ActionModelCOM_EST([None]*self.ncontact, self.m, None, None, None, self.dt)
            self.terminal = ActionModelCOM_EST([None]*self.ncontact, self.m, None, None, None, self.dt, isTerminal=True)


    def set_weights(self, xcoeff, ucoeff):
        """
        Sets the weights of the optimization problem (costs)
        xcoeff (list): tracking coefficient
        ucoeff (list): control coefficient
        """
        self.running.set_weights(xcoeff, ucoeff)
        self.terminal.set_weights(xcoeff, ucoeff)

    def set_cost_params(self, xref, r):
        """
        Sets the parameters of the optimization problem (costs)
        r (list): Contact locations
        xref (list): desired state (x, v, L)
        """
        self.running.update_params(xref, r)
        self.terminal.update_params(xref, r)

    def compute_xref(self, q, des_com, des_vcom, des_quat):
        """
        This function computes the AMOM required to correct for orientation
        q : current joint configuration
        des_com : desired com
        des_vcom : desired com velocity
        des_quat : desired orientation
        """
        xref = np.zeros(9)
        xref[0:3] = des_com
        xref[3:6] = des_vcom
        pin_quat = pin.Quaternion(np.array(q[3:7]))
        pin_des_quat = pin.Quaternion(np.array(des_quat))

        xref[6:] = self.correction_ratio * pin.log3((pin_des_quat*(pin_quat.inverse())).toRotationMatrix())

        return xref

    def optimize(self, t0, x0_hat, xref, r):
        """
        Input:
            x0_hat : current augmented state
            xref (list): desired state (x, v, L)
            r (list): Contact locations
        """
        self.set_cost_params(xref, r)

        if self.estimator == "ERSKF":
            x0 = x0_hat
        elif self.estimator == "EKF"  or self.estimator == "NA":
            x0 = x0_hat[:9]
            self.running.df = x0_hat[9:12]
            self.running.dtau = x0_hat[12:15]
            self.terminal.df = x0_hat[9:12]
            self.terminal.dtau = x0_hat[12:15]
        
        problem = crocoddyl.ShootingProblem(x0, [self.running] * self.n_col, self.terminal)

        ddp = crocoddyl.SolverDDP(problem)
        # ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
        xs = [x0] * (self.n_col + 1)
        uref = np.array([0, 0, 0] * self.ncontact)
        us = [uref] * self.n_col
        converged = ddp.solve(xs, us, maxiter=100)

        if not converged:
            print(" DDP solver DID NOT CONVERGED ".center(LINE_WIDTH, "-"))


        
        xs = ddp.xs
        us = ddp.us
        K = ddp.K
        self.K = []

        for i in range(len(us)-1):
            if i == 0:
                self.xs_int = np.linspace(xs[i], xs[i+1], int(self.dt/self.dt_ctrl))
                self.us_int = np.linspace(us[i], us[i+1], int(self.dt/self.dt_ctrl))
                self.K = [K[i]] * int(self.dt/self.dt_ctrl)
            else:
                self.xs_int = np.vstack((self.xs_int, np.linspace(xs[i], xs[i+1], int(self.dt/self.dt_ctrl))))
                self.us_int = np.vstack((self.us_int, np.linspace(us[i], us[i+1], int(self.dt/self.dt_ctrl))))
                self.K = self.K + [K[i]] * int(self.dt/self.dt_ctrl)
        return t0, self.xs_int, self.us_int, ddp.Vxx[1], ddp.Vx[1], self.K

    def optimize_parallel(self, channel):
        
        while True:
            t0, x0_hat, xref, r = channel.recv()
            t0, xs_int, us_int, Vxx, Vx, K = self.optimize(t0, x0_hat, xref, r)
            channel.send((t0, xs_int, us_int, Vxx, Vx, K))

    def plot(self):
        time = 0.001*np.arange(0, self.xs_int.shape[0])
        
        xref_traj = np.zeros((self.xs_int.shape[0], 9))
        xref_traj[:] += self.xref
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(time, np.array(self.xs_int)[:, 0])
        ax1.plot(time, xref_traj[:, 0], "-.")
        ax2.plot(time, np.array(self.xs_int)[:, 1])
        ax2.plot(time, xref_traj[:, 1], "-.")
        ax3.plot(time, np.array(self.xs_int)[:, 2])
        ax3.plot(time, xref_traj[:, 2], "-.")
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax1.set_ylabel(r"$p_x$")
        ax2.set_ylabel(r"$p_y$")
        ax3.set_ylabel(r"$p_z$")
            
        
        fig, axs = plt.subplots(4, 1, constrained_layout=True)
        for i in range(4):
            axs[i].plot(np.array(self.us_int)[:, i * 3 + 0], label="Fx")
            axs[i].plot(np.array(self.us_int)[:, i * 3 + 1], label="Fy")
            axs[i].plot(np.array(self.us_int)[:, i * 3 + 2], label="Fz")
            axs[i].grid()
            axs[i].legend()
        plt.show()