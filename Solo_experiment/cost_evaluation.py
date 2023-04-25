import numpy as np
import enum
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as m_patches
from matplotlib.collections import PatchCollection
import pickle
import sys
import crocoddyl
import pinocchio as pin

from planner import RobotCenteroidalPlanner
from robot_properties_solo.solo12wrapper import Solo12Config


DEFAULT_FONT_SIZE = 35
DEFAULT_AXIS_FONT_SIZE = DEFAULT_FONT_SIZE
DEFAULT_LINE_WIDTH = 4  # 13
DEFAULT_MARKER_SIZE = 4
DEFAULT_FONT_FAMILY = "sans-serif"
DEFAULT_FONT_SERIF = [
    "Times New Roman",
    "Times",
    "Bitstream Vera Serif",
    "DejaVu Serif",
    "New Century Schoolbook",
    "Century Schoolbook L",
    "Utopia",
    "ITC Bookman",
    "Bookman",
    "Nimbus Roman No9 L",
    "Palatino",
    "Charter",
    "serif",
]
DEFAULT_FIGURE_FACE_COLOR = "white"  # figure facecolor; 0.75 is scalar gray
DEFAULT_LEGEND_FONT_SIZE = 30  # DEFAULT_FONT_SIZE
DEFAULT_AXES_LABEL_SIZE = DEFAULT_FONT_SIZE  # fontsize of the x any y labels
DEFAULT_TEXT_USE_TEX = False
LINE_ALPHA = 0.9
SAVE_FIGURES = False
FILE_EXTENSIONS = ["pdf", "png"]  # ,'eps']
FIGURES_DPI = 150
SHOW_FIGURES = False
FIGURE_PATH = "./plot/"

mpl.rcdefaults()
mpl.rcParams["lines.linewidth"] = DEFAULT_LINE_WIDTH
mpl.rcParams["lines.markersize"] = DEFAULT_MARKER_SIZE
mpl.rcParams["patch.linewidth"] = 1
mpl.rcParams["font.family"] = DEFAULT_FONT_FAMILY
mpl.rcParams["font.size"] = DEFAULT_FONT_SIZE
mpl.rcParams["font.serif"] = DEFAULT_FONT_SERIF
mpl.rcParams["text.usetex"] = DEFAULT_TEXT_USE_TEX
mpl.rcParams["axes.labelsize"] = DEFAULT_AXES_LABEL_SIZE
mpl.rcParams["axes.grid"] = True
mpl.rcParams["legend.fontsize"] = DEFAULT_LEGEND_FONT_SIZE
# opacity of of legend frame
mpl.rcParams["legend.framealpha"] = 1.0
mpl.rcParams["figure.facecolor"] = DEFAULT_FIGURE_FACE_COLOR
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
scale = 1.0
mpl.rcParams["figure.figsize"] = 30 * scale, 10 * scale  # 23, 18  # 12, 9
line_styles = 10 * ["g", "b", "r", "c", "y", "k", "m"]


EXPERIMENT = 0  # 0, 1 or 2


if EXPERIMENT == 0:
    file_names = [
        "dropped_FILMED/ERSKF_dropped_FILMED",
        "dropped_FILMED/EKF_dropped_FILMED",
    ]
elif EXPERIMENT == 1:
    file_names = [
        "quantitative_result/ERSKF_dz_+20_FILMED",
        "quantitative_result/EKF_dz_+20_FILMED",
    ]
elif EXPERIMENT == 2:
    file_names = [
        "quantitative_result/ERSKF_dz_-10_FILMED",
        "quantitative_result/EKF_dz_-10_FILMED",
    ]

print(file_names)

labels = ["RS-EKF", "EKF"]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(40, 15))


def smooth(signal):
    avg = [np.mean(signal[i : i + 10]) for i in range(len(signal))]
    return np.array(avg)


minimum_base_pos = [0, 0]

cost_list = [[], []]
for k, file_name in enumerate(file_names):

    with open("data/" + file_name, "rb") as fp:
        b = pickle.load(fp)

    record_estimate = b[0]
    record_COM_mes = b[1]
    record_push = b[2]
    record_f = b[3]
    record_f_py = b[4]
    record_q = b[5]
    record_v = b[6]
    record_vdes = b[7]
    record_tau = b[8]
    estimator_type = b[9]

    if EXPERIMENT == 0:
        if estimator_type == "EKF":
            shift = 3650
            t_start = 5000 + shift

        elif estimator_type == "ERSKF":
            t_start = 5000
    else:
        t_start = 0

    T_tot = 6000
    t_end = t_start + T_tot

    record_COM_mes = np.array(record_COM_mes)[0][t_start:t_end]
    record_estimate = np.array(record_estimate)[0][t_start:t_end]
    record_q = np.array(record_q)[0][t_start:t_end]
    record_f = np.array(record_f)[0][t_start:t_end]

    minimum_base_pos[k] = np.min(record_q[:, 2])

    dt_ctrl = 1 * 1e-3
    pin_robot = Solo12Config.buildRobotWrapper()
    f_arr = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]

    ## init planner

    des_com = [0, 0, 0.0]
    des_vcom = [0, 0, 0]

    n_col = 4
    dt_plan = 0.05
    planner = RobotCenteroidalPlanner(pin_robot, n_col, dt_ctrl, dt_plan)
    xcoeff = [1e2, 1e2, 1e2, 1e1, 1e1, 1e1, 1e1, 1e1, 1e1]
    ucoeff = [1e-4, 1e-4, 1e-6] * len(f_arr) * 3
    planner.set_action_model("NA")
    planner.set_weights(xcoeff, ucoeff)

    for i in range(T_tot):
        x_ref = planner.compute_xref(record_q[i], des_com, des_vcom, [0, 0, 0, 1])

        planner.running.df = np.zeros(3)
        planner.running.dtau = np.zeros(3)
        planner.terminal.df = np.zeros(3)
        planner.terminal.dtau = np.zeros(3)
        if i < 1400 and EXPERIMENT == 0:
            planner.running.df = np.array([0, 0, 20])

        r = np.zeros((4, 3))  # Not used for the cost evaluation
        planner.running.update_params(x_ref, r)

        x = record_COM_mes[i]
        u = record_f[i]

        problem = crocoddyl.ShootingProblem(
            x, [planner.running] * planner.n_col, planner.terminal
        )

        problem.runningModels[0].calc(problem.runningDatas[0], x, u)
        cost_list[k].append(problem.runningDatas[0].cost)

    record_COM_mes = np.array(record_COM_mes)

    time_lin = np.linspace(0, dt_ctrl * len(record_COM_mes), len(record_COM_mes))

    esti = smooth(list(record_estimate[:, 2]))
    ax1.plot(
        time_lin,
        record_COM_mes[:, 2],
        color=line_styles[k],
        label=labels[k] + " measurement",
    )
    ax1.plot(time_lin, esti, "-.", color=line_styles[k], label=labels[k] + " estimate")
    esti = smooth(list(record_estimate[:, 9 + 2]))
    ax2.plot(time_lin, esti, "-.", color=line_styles[k], label=labels[k] + "_estimate")


cost_list = np.array(cost_list)

ax3.plot(time_lin, cost_list[0], "g", label=labels[0])
ax3.plot(time_lin, cost_list[1], "b", label=labels[1])

ax1.plot(
    time_lin, np.array([0.0] * len(time_lin)), "--", color="black"
)  # , label="Target")
if EXPERIMENT == 0:
    ax1.axvline(x=1.4, color="black")
    ax2.axvline(x=1.4, color="black")
    ax3.axvline(x=1.4, color="black")
ax1.set_ylabel(r"$p_z$ [m]")
ax2.set_ylabel(r"$F_{ext}^z [N]$")
ax3.set_ylabel(r"Cost")
ax3.set_xlabel(r"Time [s]")
ax1.set_xlim(0.0, 6.0)
ax2.set_xlim(0.0, 6.0)
ax3.set_xlim(0.0, 6.0)
ax1.set_xticklabels([])
ax2.set_xticklabels([])

ax1.legend()
plt.tight_layout()
plt.savefig(FIGURE_PATH + "zfc-plot" + ".pdf", bbox_inches="tight")


print("Delta minimum_base_pos ", minimum_base_pos[0] - minimum_base_pos[1])

rsekf = cost_list[0]
print("rs-ekf mean cost ", np.mean(rsekf))

ekf = cost_list[1]
print("ekf mean cost ", np.mean(ekf))

Gain = (np.mean(rsekf) - np.mean(ekf)) / np.mean(ekf)
print("gain ", Gain * 100)

plt.show()
