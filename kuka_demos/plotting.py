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
line_styles = 10 * ["b", "g", "r", "c", "y", "k", "m"]


dt_sim = 2e-3

# Figure 1: Cost

EKF_COST = np.load("ekf_cost.npy")
ERSKF_COST = np.load("rs-cost.npy")

EKF_COST_mean = np.mean(EKF_COST, axis=0)
ERSKF_COST_mean = np.mean(ERSKF_COST, axis=0)

EKF_COST_median = np.median(EKF_COST, axis=0)
ERSKF_COST_median = np.median(ERSKF_COST, axis=0)

EKF_COST_25 = np.percentile(EKF_COST, 25, axis=0)
ERSKF_COST_25 = np.percentile(ERSKF_COST, 25, axis=0)

EKF_COST_75 = np.percentile(EKF_COST, 75, axis=0)
ERSKF_COST_75 = np.percentile(ERSKF_COST, 75, axis=0)

T_sim = EKF_COST.shape[1] + 1
tspan_sim = np.linspace(0, T_sim * dt_sim, T_sim)

print("EKF cost", np.mean(EKF_COST_mean))
print("ERSKF cost", np.mean(ERSKF_COST_mean))

gain = 100 * (np.mean(ERSKF_COST_mean) - np.mean(EKF_COST_mean)) / np.mean(EKF_COST_mean)

print("Average COST GAIN: ", gain)

plt.figure(figsize=(21, 10))
# plt.plot(tspan_sim[:-1], EKF_COST_mean, color="b", linestyle='-', label="EKF")
# plt.plot(tspan_sim[:-1], ERSKF_COST_mean, color="g", linestyle='-', label="RS-EKF")
plt.plot(tspan_sim[:-1], EKF_COST_median, color="b", linestyle="-", label="EKF")
plt.plot(tspan_sim[:-1], ERSKF_COST_median, color="g", linestyle="-", label="RS-EKF")
plt.fill_between(tspan_sim[:-1], EKF_COST_25, EKF_COST_75, color="b", alpha=0.2)
plt.fill_between(tspan_sim[:-1], ERSKF_COST_25, ERSKF_COST_75, color="g", alpha=0.3)
plt.xlabel("Time [s]")
plt.ylabel("Cost")
plt.xlim(0, T_sim * dt_sim)
plt.legend()
plt.savefig(FIGURE_PATH + "cost_median" + ".pdf", bbox_inches="tight")


EKF_MSE_TRAJ_LIST = np.load("ekf_mse.npy")
ERSKF_MSE_TRAJ_LIST = np.load("rs-ekf_mse.npy")


# Figure 2: MSE

EKF_MSE_mean = np.mean(EKF_MSE_TRAJ_LIST, axis=0)
ERSKF_MSE_mean = np.mean(ERSKF_MSE_TRAJ_LIST, axis=0)

EKF_MSE_median = np.median(EKF_MSE_TRAJ_LIST, axis=0)
ERSKF_MSE_median = np.median(ERSKF_MSE_TRAJ_LIST, axis=0)

EKF_MSE_25 = np.percentile(EKF_MSE_TRAJ_LIST, 25, axis=0)
ERSKF_MSE_25 = np.percentile(ERSKF_MSE_TRAJ_LIST, 25, axis=0)

EKF_MSE_75 = np.percentile(EKF_MSE_TRAJ_LIST, 75, axis=0)
ERSKF_MSE_75 = np.percentile(ERSKF_MSE_TRAJ_LIST, 75, axis=0)

T_sim = EKF_MSE_TRAJ_LIST.shape[1]
tspan_sim = np.linspace(0, T_sim * dt_sim, T_sim)

fig, ax = plt.subplots(3, 1, figsize=(21, 16), sharex="col")
xyz = ["x", "y", "z"]
for i in range(3):
    # ax[i].plot(tspan_sim, EKF_MSE_mean[:,i], color="b", linestyle='-', label="mean EKF")
    # ax[i].plot(tspan_sim, ERSKF_MSE_mean[:,i], color="g", linestyle='-', label="mean RS-EKF")
    ax[i].plot(tspan_sim, EKF_MSE_median[:, i], color="b", linestyle="-", label="EKF")
    ax[i].plot(tspan_sim, ERSKF_MSE_median[:, i], color="g", linestyle="-", label="RS-EKF")
    ax[i].fill_between(tspan_sim, EKF_MSE_25[:, i], EKF_MSE_75[:, i], color="b", alpha=0.2)
    ax[i].fill_between(tspan_sim, ERSKF_MSE_25[:, i], ERSKF_MSE_75[:, i], color="g", alpha=0.3)

    ax[i].set_ylabel("$P^{EE}_%s$ " % xyz[i])
    ax[i].set_xlim(0, T_sim * dt_sim)

ax[2].set_xlabel("Time [s]")
ax[0].legend(loc="upper right")

plt.tight_layout()
plt.savefig(FIGURE_PATH + "ee-mse_error" + ".pdf", bbox_inches="tight")


# Figure 2: Cost and MSE


EKF_MSE_TRAJ_LIST = np.sum(EKF_MSE_TRAJ_LIST, axis=2)
ERSKF_MSE_TRAJ_LIST = np.sum(ERSKF_MSE_TRAJ_LIST, axis=2)

EKF_MSE_mean = np.mean(EKF_MSE_TRAJ_LIST, axis=0)
ERSKF_MSE_mean = np.mean(ERSKF_MSE_TRAJ_LIST, axis=0)

EKF_MSE_median = np.median(EKF_MSE_TRAJ_LIST, axis=0)
ERSKF_MSE_median = np.median(ERSKF_MSE_TRAJ_LIST, axis=0)

EKF_MSE_25 = np.percentile(EKF_MSE_TRAJ_LIST, 25, axis=0)
ERSKF_MSE_25 = np.percentile(ERSKF_MSE_TRAJ_LIST, 25, axis=0)

EKF_MSE_75 = np.percentile(EKF_MSE_TRAJ_LIST, 75, axis=0)
ERSKF_MSE_75 = np.percentile(ERSKF_MSE_TRAJ_LIST, 75, axis=0)


fig, ax = plt.subplots(2, 1, figsize=(21, 14), sharex="col")
xyz = ["x", "y", "z"]
ax[0].plot(tspan_sim, EKF_MSE_median[:], color="b", linestyle="-", label="EKF")
ax[0].plot(tspan_sim, ERSKF_MSE_median[:], color="g", linestyle="-", label="RS-EKF")
ax[0].fill_between(tspan_sim, EKF_MSE_25[:], EKF_MSE_75[:], color="b", alpha=0.2)
ax[0].fill_between(tspan_sim, ERSKF_MSE_25[:], ERSKF_MSE_75[:], color="g", alpha=0.3)

ax[1].plot(tspan_sim[:-1], EKF_COST_median, color="b", linestyle="-", label="EKF")
ax[1].plot(tspan_sim[:-1], ERSKF_COST_median, color="g", linestyle="-", label="RS-EKF")
ax[1].fill_between(tspan_sim[:-1], EKF_COST_25, EKF_COST_75, color="b", alpha=0.2)
ax[1].fill_between(tspan_sim[:-1], ERSKF_COST_25, ERSKF_COST_75, color="g", alpha=0.3)

ax[0].set_ylabel("MSE")
ax[1].set_ylabel("Cost")

ax[0].set_xlim(0, T_sim * dt_sim)
ax[1].set_xlim(0, T_sim * dt_sim)


ax[1].set_xlabel("Time [s]")
ax[0].legend(loc="upper right")
plt.tight_layout()
plt.savefig(FIGURE_PATH + "mse_cost" + ".pdf", bbox_inches="tight")


plt.show()
