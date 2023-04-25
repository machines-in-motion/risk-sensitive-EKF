import numpy as np
import crocoddyl
import matplotlib.pyplot as plt

LINE_WIDTH = 100


class ActionModelCOM(crocoddyl.ActionModelAbstract):
    """
    COM model
    """

    def __init__(self, r, mass, xref, xcoeff, ucoeff, dt=5e-2, isTerminal=False):
        """
        Input:
            r (list): Contact locations
            mass (float): Total mass of the robot
            xref (list): desired state (x, v, L)
            xcoeff (list): tracking coefficient
            ucoeff (list): control coefficient
            dt (float): integration time step
            isTerminal (boolean): terminal boolean
        """
        self.r = r
        self.mass = mass
        self.xref = xref
        self.xcoeff = xcoeff
        self.ucoeff = ucoeff
        self.isTerminal = isTerminal
        self.g = -9.81
        self.dt = dt
        self.nx = 9
        nu = len(r) * 3
        self.ncontact = len(r)
        state = crocoddyl.StateVector(self.nx)
        crocoddyl.ActionModelAbstract.__init__(self, state, nu)


        self.max_Fz_barrier = 10
        self.min_Fz_barrier = 0
        self.coeff_barrier = 1e5


        self.df = np.zeros(3)
        self.dtau = np.zeros(3)

    def update_params(self, xref, r):
        self.xref = xref
        self.r = r

    def set_weights(self, xcoeff, ucoeff):
        self.xcoeff = xcoeff
        self.ucoeff = ucoeff

    def dynamics(self, x, u):
        xnext = np.zeros(self.nx)
        xnext += x
        xnext[:3] += x[3:6] * self.dt
        xnext[5] += self.g * self.dt
        for i in range(self.ncontact):
            F = u[3 * i : 3 * (i + 1)]
            xnext[3:6] += F * self.dt / self.mass 
            xnext[6:] += np.cross(self.r[i] - x[:3], F) * self.dt 
    
        xnext[3:6] += self.df * self.dt / self.mass 
        xnext[6:] += self.dtau * self.dt
        return xnext

    def grav_compensation(self):
        ureg = (- self.mass * self.g - self.df[2] ) / self.ncontact
        return np.array([0, 0, ureg] * self.ncontact)

    def _running_cost(self, x, u):
        ureg_vec = self.grav_compensation()
        cost = 0
        for i in range(self.nx):
            cost += self.xcoeff[i] * (x[i] - self.xref[i]) ** 2
        for i in range(self.nu):
            cost += self.ucoeff[i] * (u[i] - ureg_vec[i]) ** 2
        for i in range(self.ncontact):
            if u[3*i+2] > self.max_Fz_barrier:
                cost += self.coeff_barrier * (u[3*i+2] - self.max_Fz_barrier) ** 2
            if u[3*i+2] < self.min_Fz_barrier:
                cost += self.coeff_barrier * (u[3*i+2] - self.min_Fz_barrier) ** 2
        return cost

    def _terminal_cost(self, x):
        cost = 0
        for i in range(self.nx):
            cost += self.xcoeff[i] * (x[i] - self.xref[i]) ** 2
        return cost

    def calc(self, data, x, u=None):
        if self.isTerminal:
            data.cost = self._terminal_cost(x)
            data.xnext = np.zeros(self.nx)
        else:
            data.cost = self._running_cost(x, u)
            data.xnext = self.dynamics(x, u)

    def calcDiff(self, data, x, u=None):
        Fx = np.eye(self.nx)
        Fu = np.zeros([self.nx, self.nu])

        Lx = np.zeros([self.nx])
        Lu = np.zeros([self.nu])
        Lxx = np.zeros([self.nx, self.nx])
        Luu = np.zeros([self.nu, self.nu])
        Lxu = np.zeros([self.nx, self.nu])

        for i in range(self.nx):
            Lx[i] = 2 * self.xcoeff[i] * (x[i] - self.xref[i])
            Lxx[i, i] = 2 * self.xcoeff[i]

        if not self.isTerminal:
            ureg_vec = self.grav_compensation()
            for i in range(self.nu):
                Lu[i] += 2 * self.ucoeff[i] * (u[i] - ureg_vec[i])
                Luu[i, i] += 2 * self.ucoeff[i]

            for i in range(self.ncontact):
                if u[3*i+2] > self.max_Fz_barrier:
                    Lu[3*i+2] += 2 * self.coeff_barrier * (u[3*i+2] - self.max_Fz_barrier) 
                    Luu[3*i+2, 3*i+2] += 2 * self.coeff_barrier

                if u[3*i+2] < self.min_Fz_barrier:
                    Lu[3*i+2] += 2 * self.coeff_barrier * (u[3*i+2] - self.min_Fz_barrier) 
                    Luu[3*i+2, 3*i+2] += 2 * self.coeff_barrier


        Fx[:3, 3:6] += self.dt * np.eye(3)

        for i in range(self.ncontact):
            F = u[3 * i : 3 * (i + 1)]

            crossMat = np.zeros((3, 3))
            crossMat[2, 1] += F[0]
            crossMat[1, 2] += -F[0]
            crossMat[0, 2] += F[1]
            crossMat[2, 0] += -F[1]
            crossMat[1, 0] += F[2]
            crossMat[0, 1] += -F[2]
            Fx[6:, :3] += crossMat * self.dt

            crossMat = np.zeros((3, 3))
            vec = self.r[i] - x[:3]
            crossMat[2, 1] += vec[0]
            crossMat[1, 2] += -vec[0]
            crossMat[0, 2] += vec[1]
            crossMat[2, 0] += -vec[1]
            crossMat[1, 0] += vec[2]
            crossMat[0, 1] += -vec[2]
            Fu[6:, 3 * i : 3 * (i + 1)] += crossMat * self.dt
            Fu[3:6, 3 * i : 3 * (i + 1)] += np.eye(3) * self.dt / self.mass

        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()


class ActionModelCOM_EST(crocoddyl.ActionModelAbstract):
    def __init__(self, r, mass, xref, xcoeff, ucoeff, dt=0.001, isTerminal=False):
        """
        Input:
            r (list): Contact locations
            mass (float): Total mass of the robot
            xref (list): desired state (x, v, L)
            xcoeff (list): tracking coefficient
            ucoeff (list): control coefficient
            dt (float): integration time step
            isTerminal (boolean): terminal boolean
        """    
        self.modelCOM = ActionModelCOM(r, mass, xref, xcoeff, ucoeff, dt, isTerminal)
        self.nx = self.modelCOM.nx + 6

        state = crocoddyl.StateVector(self.nx)
        crocoddyl.ActionModelAbstract.__init__(self, state, self.modelCOM.nu)

        self.dataCOM = self.modelCOM.createData()
        self.dt = dt

    def update_params(self, xref, r):
        self.modelCOM.update_params(xref, r)

    def set_weights(self, xcoeff, ucoeff):
        self.modelCOM.xcoeff = xcoeff
        self.modelCOM.ucoeff = ucoeff

    def calc(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)

        self.modelCOM.df = x[9:12]
        self.modelCOM.dtau = x[12:15]

        self.modelCOM.calc(self.dataCOM, x[:self.modelCOM.nx], u)
        
        xnext = x.copy()
        xnext[: self.modelCOM.nx] = self.dataCOM.xnext.copy()
        data.xnext = xnext
        data.cost = self.dataCOM.cost 

    def calcDiff(self, data, x, u=None):
        Fx = np.zeros([self.nx, self.nx])
        Fu = np.zeros([self.nx, self.nu])

        Lx = np.zeros([self.nx])
        Lu = np.zeros([self.nu])
        Lxx = np.zeros([self.nx, self.nx])
        Luu = np.zeros([self.nu, self.nu])
        Lxu = np.zeros([self.nx, self.nu])


        if u is None:
            u = np.zeros(self.nu)

        self.modelCOM.df = x[9:12]
        self.modelCOM.dtau = x[12:15]
        self.modelCOM.calcDiff(self.dataCOM, x[:self.modelCOM.nx], u)



        Fx[: self.modelCOM.nx, : self.modelCOM.nx] += self.dataCOM.Fx.copy()
        Fx[self.modelCOM.nx:, self.modelCOM.nx:] += np.eye(6)
        Fx[3:6, 9:12] += np.eye(3) * self.dt / self.modelCOM.mass
        Fx[6:9, 12:15] += np.eye(3) * self.dt

        Fu[: self.modelCOM.nx] += self.dataCOM.Fu.copy()


        Lx[: self.modelCOM.nx] = self.dataCOM.Lx.copy()
        Lxx[: self.modelCOM.nx, : self.modelCOM.nx] = self.dataCOM.Lxx.copy()

        ureg = (- self.modelCOM.mass * self.modelCOM.g - self.modelCOM.df[2] ) / self.modelCOM.ncontact
        for i in range(self.modelCOM.ncontact):
            Lx[self.modelCOM.nx + 2] +=  self.modelCOM.ucoeff[3*i + 2] * 2 * (1 / self.modelCOM.ncontact) * (u[3*i + 2] - ureg)
            Lxx[self.modelCOM.nx + 2: self.modelCOM.nx + 2] += self.modelCOM.ucoeff[3*i + 2] * 2 * (1 / self.modelCOM.ncontact) ** 2


        # Those terms do not need to be augmented as they are not use by the RS filter
        Lxu[: self.modelCOM.nx] = self.dataCOM.Lxu.copy()
        Lu = self.dataCOM.Lu.copy()
        Luu = self.dataCOM.Luu.copy()

        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()

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


        
if __name__ == "__main__":
    print(" Testing COM model with DDP ".center(LINE_WIDTH, "#"))

    T = 100
    dt = 0.05
    time = np.linspace(0, T * dt, T + 1)

    x0 = np.array([0] * 9 + [0, 0, 0] + [0] * 3)
    # x0 = np.zeros(9)

    mass = 1

    c1 = np.array([2, 2, 0])
    c2 = np.array([-2, 2, 0])
    c3 = np.array([2, -2, 0])
    c4 = np.array([-2, -2, 0])
    r = [c1, c2, c3, c4]

    ncontact = len(r)
    xref = [0.2, 0, 30, 0, 0, 0, 0, 0, 0]
    xcoeff = [1.0] * 9
    ucoeff = [1e0] * ncontact * 3

    running = ActionModelCOM_EST(r, mass, xref, xcoeff, ucoeff, dt)
    terminal = ActionModelCOM_EST(r, mass, xref, xcoeff, ucoeff, dt, isTerminal=True)
    # running = ActionModelCOM(r, mass, xref, xcoeff, ucoeff, dt)
    # terminal = ActionModelCOM(r, mass, xref, xcoeff, ucoeff, dt, isTerminal=True)

    problem = crocoddyl.ShootingProblem(x0, [running] * T, terminal)
    ddp = crocoddyl.SolverDDP(problem)
    ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
    xs = [x0] * (T + 1)
    uref = np.array([0, 0, 0] * ncontact)
    us = [uref] * T
    converged = ddp.solve(xs, us, maxiter=100)

    if converged:
        print(" DDP solver has CONVERGED ".center(LINE_WIDTH, "-"))
    else:
        print(" DDP solver DID NOT CONVERGED ".center(LINE_WIDTH, "-"))

    xref_traj = np.zeros((T + 1, 9))
    xref_traj[:] += xref

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(time, np.array(ddp.xs)[:, 0])
    ax1.plot(time, xref_traj[:, 0], "-.")
    ax2.plot(time, np.array(ddp.xs)[:, 1])
    ax2.plot(time, xref_traj[:, 1], "-.")
    ax3.plot(time, np.array(ddp.xs)[:, 2])
    ax3.plot(time, xref_traj[:, 2], "-.")
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.set_ylabel(r"$p_x$")
    ax2.set_ylabel(r"$p_y$")
    ax3.set_ylabel(r"$p_z$")

    u_reg = running.modelCOM.grav_compensation()
    uref_traj = np.zeros((T, 3 * 4))
    uref_traj += u_reg 

    fig, axs = plt.subplots(ncontact, 1)
    for i in range(ncontact):
        axs[i].plot(time[:-1], np.array(ddp.us)[:, i * 3 + 0], label="Fx")
        axs[i].plot(time[:-1], np.array(ddp.us)[:, i * 3 + 1], label="Fy")
        axs[i].plot(time[:-1], np.array(ddp.us)[:, i * 3 + 2], label="Fz")
        axs[i].plot(time[:-1], uref_traj[:, i * 3 + 0], "-.", label="Fx Ref")
        axs[i].plot(time[:-1], uref_traj[:, i * 3 + 1], "-.", label="Fy Ref")
        axs[i].plot(time[:-1], uref_traj[:, i * 3 + 2], "-.", label="Fz Ref")
        axs[i].grid()
        axs[i].legend()
    plt.show()
