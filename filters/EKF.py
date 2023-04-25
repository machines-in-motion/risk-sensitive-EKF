## This is a python implementation of EKF.
## Date : 17/09/2022
## Author : Armand Jordana

import numpy as np
import scipy.linalg as scl


class EKF:
    def __init__(self, x0_hat, P0, Q, R, process_model, measurement_model):
        self.process_model = process_model
        self.measurement_model = measurement_model
        self.intData = self.process_model.createData()
        self.state = x0_hat
        self.Q = Q
        self.R = R
        self.P = P0
        self.nx = len(self.state)

    def step(self, u_prev, y):
        # Prediction step
        self.process_model.calc(self.intData, self.state, u_prev)
        self.process_model.calcDiff(self.intData, self.state, u_prev)
        F = self.intData.Fx
        state_pred = self.intData.xnext
        P_pred = F @ self.P @ F.T + self.Q

        # Update step
        dy = y - self.measurement_model.calc(state_pred)
        C = self.measurement_model.calcDiff(state_pred)
        S = C @ P_pred @ C.T + self.R

        Lb_ = scl.cho_factor(S, lower=True)
        K_t = scl.cho_solve(Lb_, C @ P_pred)
        K = K_t.T
        # K = P_pred @ C.T @ np.linalg.inv(S)
        self.state = state_pred + K @ dy
        self.P = (np.eye(self.nx) - K @ C) @ P_pred

        self.P = (self.P + self.P.T) / 2

        return self.state.copy(), self.P.copy()
