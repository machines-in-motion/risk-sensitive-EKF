import numpy as np 

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as m_patches
from matplotlib.collections import PatchCollection
import crocoddyl 
import pinocchio as pin


DEFAULT_FONT_SIZE = 35
DEFAULT_AXIS_FONT_SIZE = DEFAULT_FONT_SIZE
DEFAULT_LINE_WIDTH = 4  # 13
DEFAULT_MARKER_SIZE = 4
DEFAULT_FONT_FAMILY = 'sans-serif'
DEFAULT_FONT_SERIF = ['Times New Roman', 'Times', 'Bitstream Vera Serif', 'DejaVu Serif', 'New Century Schoolbook',
                      'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']
DEFAULT_FIGURE_FACE_COLOR = 'white'    # figure facecolor; 0.75 is scalar gray
DEFAULT_LEGEND_FONT_SIZE = 30 #DEFAULT_FONT_SIZE
DEFAULT_AXES_LABEL_SIZE = DEFAULT_FONT_SIZE  # fontsize of the x any y labels
DEFAULT_TEXT_USE_TEX = False
LINE_ALPHA = 0.9
SAVE_FIGURES = False
FILE_EXTENSIONS = ['pdf', 'png']  # ,'eps']
FIGURES_DPI = 150
SHOW_FIGURES = False
FIGURE_PATH = './plot/'

mpl.rcdefaults()
mpl.rcParams['lines.linewidth'] = DEFAULT_LINE_WIDTH
mpl.rcParams['lines.markersize'] = DEFAULT_MARKER_SIZE
mpl.rcParams['patch.linewidth'] = 1
mpl.rcParams['font.family'] = DEFAULT_FONT_FAMILY
mpl.rcParams['font.size'] = DEFAULT_FONT_SIZE
mpl.rcParams['font.serif'] = DEFAULT_FONT_SERIF
mpl.rcParams['text.usetex'] = DEFAULT_TEXT_USE_TEX
mpl.rcParams['axes.labelsize'] = DEFAULT_AXES_LABEL_SIZE
mpl.rcParams['axes.grid'] = True
mpl.rcParams['legend.fontsize'] = DEFAULT_LEGEND_FONT_SIZE
# opacity of of legend frame
mpl.rcParams['legend.framealpha'] = 1.
mpl.rcParams['figure.facecolor'] = DEFAULT_FIGURE_FACE_COLOR
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
scale = 1.0
mpl.rcParams['figure.figsize'] = 30*scale, 10*scale #23, 18  # 12, 9
# line_styles = 10*['g-', 'r--', 'b-.', 'k:', '^c', 'vm', 'yo']
line_styles = 10*['g', 'b',  'r', 'c', 'y', 'k', 'm']

# Get frame position
def get_p_(q, model, id_endeff):
    '''
    Returns end-effector positions given q trajectory 
        q         : joint positions
        model     : pinocchio model
        id_endeff : id of EE frame
    '''
    
    data = model.createData()
    if(type(q)==np.ndarray and len(q.shape)==1):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        p = data.oMf[id_endeff].translation.T
    else:
        N = np.shape(q)[0]
        p = np.empty((N,3))
        for i in range(N):
            pin.forwardKinematics(model, data, q[i])
            pin.updateFramePlacements(model, data)
            p[i,:] = data.oMf[id_endeff].translation.T
    return p


class KukaCircle:
    def __init__(self, robot_simulator, horizon, dt, dt_sim):
        self.dt = dt
        self.dt_sim = dt_sim
        self.N_int = int(np.round(dt / dt_sim))
        self.robot_simulator = robot_simulator
        self.nq = robot_simulator.pin_robot.model.nq
        self.nv = robot_simulator.pin_robot.model.nv
        self.nu = self.nq
        self.nx = self.nq + self.nv
        self.q0 = np.array([0.1, 0.7, 0., -1., -0.5, 1.5, 0.])
        self.v0 = np.zeros(self.nv)
        self.x0 = np.concatenate([self.q0, self.v0])
        self.robot_simulator.reset_state(self.q0, self.v0)
        self.robot_simulator.forward_robot(self.q0, self.v0)
        


        # State and actuation model
        self.state = crocoddyl.StateMultibody(robot_simulator.pin_robot.model)
        self.actuation = crocoddyl.ActuationModelFull(self.state)

        # Create cost terms 
        # Control regularization cost
        self.uResidual = crocoddyl.ResidualModelControlGrav(self.state)
        self.uRegCost = crocoddyl.CostModelResidual(self.state, self.uResidual)
        # State regularization cost
        self.xResidual = crocoddyl.ResidualModelState(self.state, self.x0)
        self.xRegCost = crocoddyl.CostModelResidual(self.state, self.xResidual)
        # endeff frame translation cost
        self.endeff_frame_id = robot_simulator.pin_robot.model.getFrameId("contact")




        ee_pos = get_p_(self.q0, self.robot_simulator.pin_robot.model, self.endeff_frame_id)
        Ttot = 12000
        self.target = np.zeros((Ttot, 3))
        self.target[:, 0] = np.array([0.1 * np.sin(i*self.dt_sim*2*np.pi/4) for i in range(Ttot)])               
        self.target[:, 1] =  np.array([0.1 * (np.cos(i*self.dt_sim*2*np.pi/4) - 1) for i in range(Ttot)])  
        self.target += ee_pos


        self.T = horizon


        # Create model for EKF
        runningCostModel = crocoddyl.CostModelSum(self.state)
        running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, runningCostModel)        
        self.runningModelfilt = crocoddyl.IntegratedActionModelEuler(running_DAM, dt_sim)


        # Running and terminal cost models
        runningModels = []
        for i in range(self.T):
            runningCostModel = crocoddyl.CostModelSum(self.state)
            runningCostModel.addCost("stateReg", self.xRegCost, 1e-2)
            runningCostModel.addCost("ctrlRegGrav", self.uRegCost, 1e-4)

            endeff_translation = self.target[i*self.N_int]
            
            frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.state, self.endeff_frame_id, endeff_translation)

            frameTranslationCost = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
            runningCostModel.addCost("translation", frameTranslationCost, 100)

            running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, runningCostModel)
            runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, self.dt)
            runningModels.append(runningModel)

        terminalCostModel = crocoddyl.CostModelSum(self.state)
        terminalCostModel.addCost("stateReg", self.xRegCost, 1e-2)
        endeff_translation = self.target[self.T*self.N_int]
        frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.state, self.endeff_frame_id, endeff_translation)
        frameTranslationCost = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
        terminalCostModel.addCost("translation", frameTranslationCost, 100)
        terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, terminalCostModel)
        terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, self.dt)

        problem = crocoddyl.ShootingProblem(self.x0, runningModels, terminalModel)
        self.ddp_solver = crocoddyl.SolverFDDP(problem)

    def update_reference(self, x0, t):
        self.ddp_solver.problem.x0 = x0

        models = list(self.ddp_solver.problem.runningModels) + [self.ddp_solver.problem.terminalModel]
        for i, m in enumerate(models):
            m.differential.costs.costs['translation'].cost.residual.reference = self.target[t+i*self.N_int]
  


    def plot_state(self, xtraj, xest, xtraj_rs, xest_rs):

        traj_q = xtraj[:,:self.nq]
        traj_v = xtraj[:,self.nv:]
        est_q = xest[:,:self.nq]
        est_v = xest[:,self.nv:]
        traj_rs_q = xtraj_rs[:,:self.nq]
        traj_rs_v = xtraj_rs[:,self.nv:]
        est_rs_q = xest_rs[:,:self.nq]
        est_rs_v = xest_rs[:,self.nv:]

        T_sim = len(xtraj)

        tspan_sim = np.linspace(0, T_sim*self.dt_sim, T_sim)
        tspan = np.linspace(0, self.T*self.dt, self.T+1)
        fig, ax = plt.subplots(self.nq, 2, sharex='col') 
        for i in range(self.nq):
            # Plot positions
            ax[i,0].plot(tspan_sim, traj_q[:,i], color=line_styles[1], linestyle='-', label='Neutral Simulation')  
            ax[i,0].plot(tspan_sim, est_q[:,i],  color=line_styles[1], linestyle=':', label='Neutral Estimate')  
            ax[i,0].plot(tspan_sim, traj_rs_q[:,i],color=line_styles[0], linestyle='-', label='RS Simulation')  
            ax[i,0].plot(tspan_sim, est_rs_q[:,i], color=line_styles[0], linestyle=':', label='RS Estimate')  


            ax[i,0].set_ylabel('$q_%s$'%i, fontsize=16)
            ax[i,0].grid(True)
            # Plot velocities
            ax[i,1].plot(tspan_sim, traj_v[:,i], linestyle='-', label='Neutral Simulation') 
            ax[i,1].plot(tspan_sim, est_v[:,i], linestyle=':', label='Neutral Estimate') 
            ax[i,1].plot(tspan_sim, traj_rs_v[:,i], linestyle='-', label='RS Simulation') 
            ax[i,1].plot(tspan_sim, est_rs_v[:,i], linestyle=':', label='RS Estimate') 

            # Labels, tick labels and grid
            ax[i,1].set_ylabel('$v_%s$'%i, fontsize=16)
            ax[i,1].grid(True)  
        # Common x-labels + align
        ax[-1,0].set_xlabel('time [s]', fontsize=16)
        ax[-1,1].set_xlabel('time [s]', fontsize=16)
        fig.align_ylabels(ax[:, 0])
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', prop={'size': 16})
        fig.align_ylabels()
        fig.suptitle('State trajectories', size=18)


    def plot_control(self, utraj, utraj_rs):
        T_sim = len(utraj) 
        # Plot Control
        tspan_sim = np.linspace(0, T_sim*self.dt_sim, T_sim)
        tspan = np.linspace(0, self.T*self.dt, self.T)
        fig, ax = plt.subplots(self.nu, 1, sharex='col') 

        for i in range(self.nu):
            ax[i].plot(tspan_sim, utraj[:,i], linestyle='-', label='Neutral Simulation')  
            ax[i].plot(tspan_sim, utraj_rs[:,i], linestyle='-', label='RS Simulation')  
            ax[i].set_ylabel('$u_%s$'%i, fontsize=16)
            ax[i].grid(True)

        # Common x-labels + align
        ax[-1].set_xlabel('time [s]', fontsize=16)
        fig.align_ylabels(ax[:])
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', prop={'size': 16})
        fig.align_ylabels()
        fig.suptitle('Control trajectories', size=18)

    
    def plot_endeff(self, xtraj, xest, xtraj_rs, xest_rs):
        traj_q = xtraj[:,:self.nq]
        est_q = xest[:,:self.nq]
        traj_rs_q = xtraj_rs[:,:self.nq]
        est_rs_q = xest_rs[:,:self.nq]

        T_sim = len(xtraj)
        tspan_sim = np.linspace(0, T_sim*self.dt_sim, T_sim)

        lin_pos_ee_traj = get_p_(traj_q, self.robot_simulator.pin_robot.model, self.endeff_frame_id)
        lin_pos_ee_est = get_p_(est_q, self.robot_simulator.pin_robot.model, self.endeff_frame_id)
        lin_pos_ee_traj_rs = get_p_(traj_rs_q, self.robot_simulator.pin_robot.model, self.endeff_frame_id)
        lin_pos_ee_est_rs = get_p_(est_rs_q, self.robot_simulator.pin_robot.model, self.endeff_frame_id)


        # fig, ax = plt.subplots(3, 1, figsize=(21, 16), sharex='col')
        # xyz = ['x', 'y', 'z']
        # for i in range(3):
        #     ax[i].plot(tspan_sim, self.target[:T_sim,i], linestyle='--', color='k', marker=None)
        #     # Plot EE position in WORLD frame 
        #     ax[i].plot(tspan_sim, lin_pos_ee_traj[:,i], color=line_styles[1], linestyle='-', label="EKF measurement")
        #     ax[i].plot(tspan_sim, lin_pos_ee_est[:,i], color=line_styles[1], linestyle=':', label="EKF estimate")
        #     ax[i].plot(tspan_sim, lin_pos_ee_traj_rs[:,i], color=line_styles[0], linestyle='-', label="RS-EKF measurement")
        #     ax[i].plot(tspan_sim, lin_pos_ee_est_rs[:,i], color=line_styles[0], linestyle=':', label="RS-EKF estimate")
        #     # Plot EE target frame translation in WORLD frame
        #     # Labels, tick labels, grid
        #     ax[i].set_ylabel('$P^{EE}_%s$ [m]'%xyz[i])
        #     ax[i].set_xlim(0, T_sim*self.dt_sim)

        #     ax[i].axvline(x = 1, color = 'black')
        #     ax[i].axvline(x = 2, color = 'black')


        # #x-label + align
        # # fig.align_ylabels(ax[:])
        # ax[i].set_xlabel('Time [s]')

        # ax[0].legend(loc='upper right')
        # plt.text(1.07, 0.91, 'Unexpected push', dict(size=30))
        # # fig.suptitle('End Eff trajectories', size=18)
        # # plt.tight_layout()

        # plt.savefig(FIGURE_PATH+"ee-plot"+".pdf", bbox_inches='tight')
        # plt.show()

        EKF_MSE = np.mean((lin_pos_ee_traj - self.target[:T_sim])**2)
        ERSKF_MSE = np.mean((lin_pos_ee_traj_rs - self.target[:T_sim])**2)
        
        # import pdb; pdb.set_trace()
        return EKF_MSE, ERSKF_MSE, np.abs(lin_pos_ee_traj - self.target[:T_sim])**2, np.abs(lin_pos_ee_traj_rs - self.target[:T_sim])**2, tspan_sim


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



if __name__ == "__main__":
    from robot_properties_kuka.iiwaWrapper import IiwaRobot
    from bullet_utils.env import BulletEnvWithGround
    import pybullet as p

    dt_sim = 1e-3
    env = BulletEnvWithGround(p.GUI, dt=dt_sim)
    # env = BulletEnvWithGround(p.DIRECT, dt=dt_sim)
    robot_simulator = IiwaRobot()
    horizon = 20
    kukacircle = KukaCircle(robot_simulator, horizon, dt=0.05, dt_sim=dt_sim)
    env.add_robot(robot_simulator)

    x0 = kukacircle.x0
    q0 = kukacircle.q0
    v0 = kukacircle.v0
    robot_simulator.reset_state(q0, v0)
    robot_simulator.forward_robot(q0, v0)
    print("[PyBullet] Created robot (id = "+str(robot_simulator.robotId)+")")

    nx = kukacircle.nx
    T = kukacircle.T

    

    # Warm start : initial state + gravity compensation
    xs_init = [x0 for i in range(T+1)]
    us_init = [np.zeros(kukacircle.nq)] * T


    #### Neutral simulation
    xtraj = [x0]
    xest = [x0]
    utraj = []
    xs = xs_init.copy()
    us = us_init.copy()
    T_sim = int(2/dt_sim)
    for t in range(T_sim-1):
        print ("\r Neutral Simulation... {} / ".format(t+1) +str(T_sim), end="")
        # Solve with DDP solver



        ddp_solver = kukacircle.create_solver(x0, t)
        # ddp_solver.setCallbacks([crocoddyl.CallbackLogger(),
        #                 crocoddyl.CallbackVerbose()])


        ddp_solver.solve(xs, us, maxiter=20, isFeasible=False)
        u = ddp_solver.us[0]
        robot_simulator.send_joint_command(u)
        env.step()
        # Measure new state from simulator 
        q, v = robot_simulator.get_state()
        # Update pinocchio model
        robot_simulator.forward_robot(q, v)
        # Record data 
        x = np.concatenate([q, v]).T
        xtraj.append(x)
        utraj.append(u)
        x0 = x
        
        #Update warm start        
        xs = list(ddp_solver.xs[1:]) + [ddp_solver.xs[-1]]
        xs[0] = x0
        us = list(ddp_solver.us[1:]) + [ddp_solver.us[-1]] 

    print(q)

    xtraj = np.array(xtraj)
    traj_q = xtraj[:, :7]
    target = kukacircle.target

    tspan_sim = np.linspace(0, T_sim*0.001, T_sim)

    lin_pos_ee_traj = get_p_(traj_q, kukacircle.robot_simulator.pin_robot.model, kukacircle.endeff_frame_id)


    # Plots
    fig, ax = plt.subplots(3, 1, figsize=(20, 15), sharex='col')
    xyz = ['x', 'y', 'z']
    for i in range(3):
        ax[i].plot(tspan_sim, target[:T_sim,i], linestyle='--', color='k', marker=None)
        ax[i].plot(tspan_sim, lin_pos_ee_traj[:,i], linestyle='-')
        ax[i].set_ylabel('$P^{EE}_%s$ [m]'%xyz[i])
    # fig.align_ylabels(ax[:])
    ax[i].set_xlabel('Time [s]')
    ax[0].legend(loc='lower right')
    # fig.suptitle('End Eff trajectories', size=18)

    plt.show()