import itertools
import os
import numpy as np
import pinocchio as pin
import time
import pybullet as p
from bullet_utils.env import BulletEnvWithGround

from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
from dynamic_graph_head import ThreadHead, SimHead, SimVicon, HoldPDController
from dg_rekf_controller import DGHeadERSKFController


## robot config and init
f_arr = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]

## init planner
n_col = 4
planner_dt = 0.05

xcoeff = [1e2, 1e2, 1e2, 1e1, 1e1, 1e1, 1e1, 1e1, 1e1]
ucoeff = [1e-4, 1e-4, 1e-6] * len(f_arr) * 3
correction_ratio = 0.2

# desired centor of mass
des_com = [0.0, 0, 0.2]
des_vcom = [0,0,0]

## EFK
filter_dt = 0.005


P0 = np.diag([1e-4]*9 +  [1e-2] *3 + [1e-2]*3)
Q = np.diag([1e-4]*6 + [1e-4]*3 + [1e-2]*3 + [1e-3]*3)
R = np.diag([1e-4]*3 + [5e-2]*3 + [1e-4]*3 )
sigma = 6 # 10 is the limit in simulation

# replan time
replan_time = 0.02
filter_replan_time = filter_dt
filter_type = "ERSKF"

run_sim = False
pin_robot = Solo12Config.buildRobotWrapper()
## init robot
if run_sim:
    dt_sim = 1e-3
    env = BulletEnvWithGround(dt=dt_sim)
    robot = Solo12Robot()
    robot = env.add_robot(robot)
    robot_config = Solo12Config()

    q0 = np.array(robot_config.initial_configuration)
    v0 = np.array(robot_config.initial_velocity)

    robot.reset_state(q0, v0)


    head = SimHead(robot, vicon_name='solo12')
    thread_head = ThreadHead(
        0.001, # dt.
        HoldPDController(head, 3., 0.05, True), # Safety controllers.
        head, # Heads to read / write from.
        [     # Utils.
            ('vicon', SimVicon(['solo12/solo12']))
        ], 
        env # Environment to step.
    )

else:
    import dynamic_graph_manager_cpp_bindings
    from dynamic_graph_head import ThreadHead, Vicon, HoldPDController
    
    head = dynamic_graph_manager_cpp_bindings.DGMHead(Solo12Config.dgm_yaml_path)

    # Create the controllers.
    hold_pd_controller = HoldPDController(head, 3., 0.05, with_sliders=True)

    thread_head = ThreadHead(
        0.001,
        hold_pd_controller,
        head,
        [
            ('vicon', Vicon('172.24.117.119:801', ['solo12/solo12']))
        ]
    )

# log function
def unique_file(basename, ext):
    basename = os.path.join("data", basename)
    actualname = "%s.%s" % (basename, ext)
    c = itertools.count()
    while os.path.exists(actualname):
        actualname = "%s_%d.%s" % (basename, next(c), ext)
    return actualname


## setup ctrl
robot_config = Solo12Config()


class Experiment_Parameters:
    def __init__(self):
        self.reset()

    def reset(self):
        self.P0 = P0 
        self.Q = Q 
        self.R = R
        self.sigma = sigma
        self.xcoeff = xcoeff
        self.ucoeff = ucoeff
        self.correction_ratio = correction_ratio




exp_params = Experiment_Parameters()


if run_sim:
    erskfctrl = DGHeadERSKFController(head, 'solo12/solo12', robot_config, pin_robot, f_arr, n_col, run_sim)
    erskfctrl.init_planner(planner_dt, xcoeff, ucoeff, des_com, des_vcom, correction_ratio)
    erskfctrl.init_filter(filter_dt, P0, Q, R, sigma, filter_type)
    erskfctrl.set_replan_times(replan_time, filter_replan_time)
    thread_head.switch_controllers(erskfctrl)
    thread_head.sim_run(10000)

else:
    thread_head.start()



def switch(name="test"):
    # Set parameters
    erskfctrl = DGHeadERSKFController(head, 'solo12/solo12', robot_config, pin_robot, f_arr, n_col)
    erskfctrl.init_planner(planner_dt, exp_params.xcoeff, exp_params.ucoeff, des_com, des_vcom, exp_params.correction_ratio)
    erskfctrl.init_filter(filter_dt, exp_params.P0, exp_params.Q, exp_params.R, exp_params.sigma, filter_type)
    erskfctrl.set_replan_times(replan_time, filter_replan_time)

    # log name
    log_name = unique_file(name, "mds")

    # Switch controller
    thread_head.switch_controllers(erskfctrl)
    thread_head.start_logging(30, log_name)


def pd_control():
    thread_head.switch_controllers(hold_pd_controller)

