import numpy as np
import crocoddyl
import matplotlib.pyplot as plt

LINE_WIDTH = 100


class QuadrotorDynamics:
    def __init__(self):
        self.g = 9.81
        self.m = 1
        self.d = 0.4
        self.nq = 3
        self.nv = 3
        self.ndx = 3
        self.nx = self.nq + self.nv
        self.nu = 2

    def nonlinear_dynamics(self, x, u):
        x_ddot = -(u[0] + u[1]) * np.sin(x[2]) / self.m
        y_ddot = (u[0] + u[1]) * np.cos(x[2]) / self.m - self.g
        th_ddot = (u[0] - u[1]) / (self.m * self.d)
        return np.array([x_ddot, y_ddot, th_ddot])


class DifferentialActionModelQuadrotor(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, xdes, ydes, dt=5e-2, isTerminal=False):
        self.dynamics = QuadrotorDynamics()
        state = crocoddyl.StateVector(self.dynamics.nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, state, self.dynamics.nu, self.dynamics.ndx
        )
        self.ndx = self.state.ndx
        self.isTerminal = isTerminal
        self.d = self.dynamics.d
        self.g = self.dynamics.g
        self.dt = dt

        self.xdes, self.ydes = xdes, ydes

        self.xcoeff = [1e2, 1e2, 1e1, 1e-2, 1e-2, 1e-2]
        self.xcoeff_ter = [1e2, 1e2, 1e1, 1e-2, 1e-2, 1e-2]
        self.ucoeff = [1e-1, 1e-1]


    def _running_cost(self, x, u):
        cost = self.xcoeff[0] * (x[0] - self.xdes) ** 2 + self.xcoeff[1] * (x[1]-self.ydes) ** 2 
        cost += self.xcoeff[2] * x[2] ** 2 + self.xcoeff[3] * x[3] ** 2 + self.xcoeff[4] * x[4] ** 2 + self.xcoeff[5] * x[5] ** 2
        cost += self.ucoeff[0] * (u[0] - self.dynamics.m * self.g / 2) ** 2 + self.ucoeff[1] * (u[1] - self.dynamics.m * self.g / 2) ** 2
        return cost

    def _terminal_cost(self, x, u):
        cost = self.xcoeff_ter[0] * (x[0] - self.xdes) ** 2 + self.xcoeff_ter[1] * (x[1]-self.ydes) ** 2  + self.xcoeff_ter[2] * x[2] ** 2
        cost += self.xcoeff_ter[3] * x[3] ** 2 + self.xcoeff_ter[4] * x[4] ** 2 + self.xcoeff_ter[5] * x[5] ** 2
        return cost

    def calc(self, data, x, u=None):

        if u is None:
            u = np.zeros(self.nu)

        data.xout = self.dynamics.nonlinear_dynamics(x, u)
        if self.isTerminal:
            data.cost = self._terminal_cost(x, u) 
        else:
            data.cost = self._running_cost(x, u)

    def calcDiff(self, data, x, u=None):
        Fx = np.zeros([3, 6])
        Fu = np.zeros([3, 2])

        Lx = np.zeros([6])
        Lu = np.zeros([2])
        Lxx = np.zeros([6, 6])
        Luu = np.zeros([2, 2])
        Lxu = np.zeros([6, 2])
        if self.isTerminal:
            Lx[0] = 2 * self.xcoeff_ter[0]  * (x[0] - self.xdes)
            Lx[1] = 2 * self.xcoeff_ter[1]  * (x[1] - self.ydes)
            Lx[2] = 2 * self.xcoeff_ter[2]  * x[2]
            Lxx[0, 0] = 2 * self.xcoeff_ter[0] 
            Lxx[1, 1] = 2 * self.xcoeff_ter[1] 
            Lxx[2, 2] = 2 * self.xcoeff_ter[2] 
            Lx[3] = 2 * self.xcoeff_ter[3] * x[3]
            Lx[4] = 2 * self.xcoeff_ter[4] * x[4]
            Lx[5] = 2 * self.xcoeff_ter[5] * x[5]
            Lxx[3, 3] = 2 * self.xcoeff_ter[3]
            Lxx[4, 4] = 2 * self.xcoeff_ter[4]
            Lxx[5, 5] = 2 * self.xcoeff_ter[5]
        else:
            Lx[0] = 2 * self.xcoeff[0] * (x[0] - self.xdes)
            Lx[1] = 2 * self.xcoeff[1] * (x[1] - self.ydes)
            Lx[2] = 2 * self.xcoeff[2] * x[2]
            Lxx[0, 0] = 2 * self.xcoeff[0]
            Lxx[1, 1] = 2 * self.xcoeff[1]
            Lxx[2, 2] = 2 * self.xcoeff[2]
            Lx[3] = 2 * self.xcoeff[3] * x[3]
            Lx[4] = 2 * self.xcoeff[4] * x[4]
            Lx[5] = 2 * self.xcoeff[5] * x[5]
            Lxx[3, 3] = 2 * self.xcoeff[3]
            Lxx[4, 4] = 2 * self.xcoeff[4]
            Lxx[5, 5] = 2 * self.xcoeff[5]
            #
            Lu[0] = 2 * self.ucoeff[0] * (u[0] - self.dynamics.m * self.g / 2)
            Lu[1] = 2 * self.ucoeff[1] * (u[1] - self.dynamics.m * self.g / 2)
            Luu[0, 0] = 2 * self.ucoeff[0]
            Luu[1, 1] = 2 * self.ucoeff[1]
            #
        Fu[0, 0] = -np.sin(x[2]) / self.dynamics.m
        Fu[0, 1] = -np.sin(x[2]) / self.dynamics.m
        #
        Fu[1, 0] = np.cos(x[2]) / self.dynamics.m
        Fu[1, 1] = np.cos(x[2]) / self.dynamics.m
        #
        Fu[2, 0] = 1 / (self.dynamics.m * self.d)
        Fu[2, 1] = -1 / (self.dynamics.m * self.d)
        #
        Fx[0, 2] = -(u[0] + u[1]) * np.cos(x[2]) / self.dynamics.m
        Fx[1, 2] = -(u[0] + u[1]) * np.sin(x[2]) / self.dynamics.m

        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()



class ActionModelQuadrotorMass(crocoddyl.ActionModelAbstract):
    def __init__(self, xdes, ydes, dt=5e-2, isTerminal=False):
        self.diffModel = DifferentialActionModelQuadrotor(xdes, ydes,
            dt=dt, isTerminal=isTerminal
        )
        self.nx = self.diffModel.dynamics.nx + 1
        self.nq = self.diffModel.dynamics.nq
        state = crocoddyl.StateVector(self.nx)
        crocoddyl.ActionModelAbstract.__init__(self, state, self.diffModel.dynamics.nu)

        self.diffData = self.diffModel.createData()
        self.isTerminal = isTerminal
        self.m = self.diffModel.dynamics.m
        self.d = self.diffModel.dynamics.d
        self.g = self.diffModel.dynamics.g
        self.dt = dt

    def calc(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)


        self.diffModel.dynamics.m = x[-1]
        self.diffModel.calc(self.diffData, x[:-1], u)
        xout = self.diffData.xout.copy()
        
        xnext = x.copy()
        xnext[: self.nq] += self.dt * x[self.nq : 2 * self.nq] + self.dt**2 * xout
        xnext[self.nq : 2 * self.nq] += self.dt * xout
        data.xnext = xnext
        
        if self.isTerminal:
            data.cost = self.diffData.cost 
        else:
            data.cost = self.diffData.cost * self.dt

    def calcDiff(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)

        mass = x[-1]
        self.diffModel.dynamics.m = mass
        self.diffModel.calcDiff(self.diffData, x[:-1], u)

        Fx = np.zeros([7, 7])
        Fu = np.zeros([7, 2])

        Lx = np.zeros([7])
        Lu = np.zeros([2])
        Lxx = np.zeros([7, 7])
        Luu = np.zeros([2, 2])
        Lxu = np.zeros([7, 2])


        Fm = np.array(
            [
                (u[0] + u[1]) * np.sin(x[2]) / mass ** 2,
                -(u[0] + u[1]) * np.cos(x[2]) / mass ** 2,
                -(u[0] - u[1]) / (mass ** 2 * self.d),
            ]
        )


        diff_Fx = self.diffData.Fx.copy()
        diff_Fu = self.diffData.Fu.copy()

        Fx += np.eye(self.nx)
        Fx[: self.nq, self.nq : 2 * self.nq] += self.dt * np.eye(self.nq)
        Fx[: self.nq, : 2 * self.nq] +=  self.dt**2 * diff_Fx
        Fx[self.nq : 2 * self.nq, : 2 * self.nq] += self.dt * diff_Fx
        Fx[: self.nq, -1] += self.dt**2 * Fm 
        Fx[self.nq : 2 * self.nq, -1] += self.dt * Fm 

        Fu[self.nq : 2 * self.nq] += self.dt * diff_Fu
        Fu[:self.nq] += self.dt**2 * diff_Fu

        Lx[:-1] = self.diffData.Lx.copy()
        Lx[-1] = - (2 * self.diffModel.ucoeff[0] * (u[0] - mass * self.g / 2)  + 2 * self.diffModel.ucoeff[1] * (u[1] - mass * self.g / 2) ) * self.g / 2

        Lxx[:-1, :-1] = self.diffData.Lxx.copy()
        Lxx[-1, -1] = 2  * (self.diffModel.ucoeff[0] + self.diffModel.ucoeff[1]) * (self.g / 2) ** 2


        Lxu[:-1] = self.diffData.Lxu.copy()
        Lxu[-1, 0] = 2  * (self.diffModel.ucoeff[0] + self.diffModel.ucoeff[1])  * self.g / 2
        Lxu[-1, 1] = 2  * (self.diffModel.ucoeff[0] + self.diffModel.ucoeff[1])  * self.g / 2

        Lu = self.diffData.Lu.copy()
        Luu = self.diffData.Luu.copy()

        if not self.isTerminal:
            Lx = Lx * self.dt
            Lu = Lu * self.dt
            Lxx = Lxx * self.dt
            Luu = Luu * self.dt
            Lxu = Lxu * self.dt


        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()


class measurement_model_full:
    def __init__(self, nx):
        self.nx = nx

    def calc(self, x):
        return x

    def calcDiff(self, x):
        return np.eye(self.nx)


class measurement_model:
    def __init__(self, nx, ncut):
        self.nx = nx
        self.ncut = ncut
        self.C = np.zeros((ncut, nx))
        self.C[:ncut, :ncut] = np.eye(ncut)

    def calc(self, x):
        return x[: self.ncut]

    def calcDiff(self, x):
        return self.C


class QuadrotorEnv:
    def __init__(self, quadrotor_model, measurement_model, P0, R):
        self.quadrotor_model = quadrotor_model
        self.measurement_model = measurement_model
        self.intData = self.quadrotor_model.createData()
        self.cov = R
        self.P0 = P0

    def init_state(self, x):
        self.state = x + np.random.multivariate_normal(np.zeros(len(x)), self.P0)
        return self.state.copy()

    def step(self, u):
        self.quadrotor_model.calc(self.intData, self.state, u)
        self.state = self.intData.xnext
        return self.state.copy()

    def obs(self):
        return self.measurement_model.calc(
            self.state.copy()
        ) + np.random.multivariate_normal(np.zeros(3), self.cov)



class PushQuadrotorEnv:
    def __init__(self, quadrotor_model, measurement_model):
        self.quadrotor_model = quadrotor_model
        self.measurement_model = measurement_model

        self.intData = self.quadrotor_model.createData()

    def init_state(self, x):
        self.state = x 
        return self.state.copy()

    def step(self, u):
        self.quadrotor_model.calc(self.intData, self.state, u)
        self.state = self.intData.xnext
        return self.state.copy()

    def obs(self):
        return self.measurement_model.calc(self.state.copy()) 

    def push(self, force):
        dx_dot = force * 0.05
        dx = force * 0.05 ** 2
        self.state[1] += dx
        self.state[4] += dx_dot




class MassQuadrotorEnv:
    def __init__(self, quadrotor_model, measurement_model):
        self.quadrotor_model = quadrotor_model
        self.measurement_model = measurement_model
        self.intData = self.quadrotor_model.createData()

    def init_state(self, x):
        self.state = x
        return self.state.copy()

    def step(self, u):
        self.quadrotor_model.calc(self.intData, self.state, u)
        self.state = self.intData.xnext
        return self.state.copy()

    def obs(self):
        return self.measurement_model.calc(self.state.copy())

    def change_mass(self, m):
        self.state[-1] = m

if __name__ == "__main__":
    print(" Testing Quadrotor with DDP ".center(LINE_WIDTH, "#"))

    T = 100
    dt = 0.05
    x0 = np.array([0, 0, 0, 0, 0, 0])

    xdes_traj = [ - np.cos(np.pi * t / T) + 1  for t in range(T+1)]
    ydes_traj = [ np.sin(np.pi * t / T)  for t in range(T+1)]


    runnings = []
    for i in range(T):
        quadrotor_diff_running = DifferentialActionModelQuadrotor(xdes_traj[i], ydes_traj[i])
        quadrotor_running = crocoddyl.IntegratedActionModelEuler(quadrotor_diff_running, dt)
        runnings.append(quadrotor_running)

    quadrotor_diff_terminal = DifferentialActionModelQuadrotor(xdes_traj[T], ydes_traj[T], isTerminal=True)

    quadrotor_terminal = crocoddyl.IntegratedActionModelEuler(quadrotor_diff_terminal, dt)
    print(" Constructing integrated models completed ".center(LINE_WIDTH, "-"))

    problem = crocoddyl.ShootingProblem(x0, runnings, quadrotor_terminal)
    print(" Constructing shooting problem completed ".center(LINE_WIDTH, "-"))

    ddp = crocoddyl.SolverDDP(problem)
    print(" Constructing DDP solver completed ".center(LINE_WIDTH, "-"))
    ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
    xs = [x0] * (T + 1)
    us = [np.zeros(2)] * T
    converged = ddp.solve(xs, us, maxiter=1000)

    if converged:
        print(" DDP solver has CONVERGED ".center(LINE_WIDTH, "-"))
    else:
        print(" DDP solver DID NOT CONVERGED ".center(LINE_WIDTH, "-"))

    plt.figure()
    plt.plot(np.array(ddp.xs)[:, 0], np.array(ddp.xs)[:, 1], label="ddp")
    plt.xlabel(r"$p_x$ [m]")
    plt.ylabel(r"$p_y$ [m]")
    plt.legend()
    plt.grid()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(np.array(ddp.xs)[:, 0], label="x")
    ax2.plot(np.array(ddp.xs)[:, 1], label="y")
    ax1.plot(np.array(xdes_traj), "-.")
    ax2.plot(np.array(ydes_traj), "-.")
    ax3.plot(np.array(ddp.xs)[:, 2], label="theta")
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.set_ylabel(r"$p_x$")
    ax2.set_ylabel(r"$p_y$")
    ax3.set_ylabel(r"$\theta$")

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(np.array(ddp.us)[:, 0], label="F1")
    ax2.plot(np.array(ddp.us)[:, 1], label="F2")
    ax1.grid()
    ax2.grid()
    ax1.set_ylabel(r"$u_1$")
    ax2.set_ylabel(r"$u_2$")
    plt.show()
