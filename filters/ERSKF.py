## This is a python implementation of RS-EKF.
## Date : 17/09/2022
## Author : Armand Jordana

import numpy as np
import scipy.linalg as scl
import eigenpy


class ERSKF:
    def __init__(self, x0_hat, P0, Q, R, sigma, process_model, measurement_model):
        self.process_model = process_model
        self.measurement_model = measurement_model
        self.intData = self.process_model.createData()
        self.state = x0_hat
        self.Q = Q
        self.R = R
        self.P = P0
        self.sigma = sigma
        self.nx = len(self.state)

    def step(self, u_prev, y, V, v):
        # Prediction step
        self.process_model.calc(self.intData, self.state, u_prev)
        self.process_model.calcDiff(self.intData, self.state, u_prev)
        F = self.intData.Fx
        state_pred = self.intData.xnext
        P_pred = F @ self.P @ F.T + self.Q

        # Update step 1
        dy = y - self.measurement_model.calc(state_pred)
        C = self.measurement_model.calcDiff(state_pred)
        S = C @ P_pred @ C.T + self.R
        Lb_ = scl.cho_factor(S, lower=True)
        K_t = scl.cho_solve(Lb_, C @ P_pred.T)
        K = K_t.T
        self.P = (np.eye(self.nx) - K @ C) @ P_pred
        self.P = (self.P + self.P.T) / 2

        mu_hat = K @ dy

        # Update step 2
        Lb = np.eye(self.nx) - self.sigma * self.P.copy() @ V

        if (np.linalg.eig(Lb)[0] < 0).any():
            raise AssertionError(
                "Sigma is too large, the shift matrix in not positive definite",
                np.linalg.eig(Lb)[0],
            )

        p2 = np.linalg.solve(Lb, mu_hat + self.sigma * self.P.copy() @ v)
        self.state = state_pred + p2

        return self.state.copy(), self.P.copy()
