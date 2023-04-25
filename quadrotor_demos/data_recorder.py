import numpy as np
import enum
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as m_patches
from matplotlib.collections import PatchCollection

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


class DataRecorder:
    def __init__(self, estimator_type, xdes_ref, ydes_ref, dt=1e-3):
        self.dt = dt
        self.estimator_type = estimator_type
        self.record_estimate = []
        self.record_state = []
        self.record_control = []
        self.record_cost = []

        self.label = []
        for i in range(len(estimator_type)):
            if estimator_type[i] == 0:
                self.label.append("EKF")
            else:
                self.label.append("RS-EKF")

        self.xdes_ref = xdes_ref
        self.ydes_ref = ydes_ref
        self.ctr = -1

    def add_experiment(self):

        self.ctr += 1

        self.record_estimate.append([])
        self.record_state.append([])
        self.record_control.append([])
        self.record_cost.append([])

    def record_data(self, est, x, u, cost):

        self.record_estimate[self.ctr].append(est)
        self.record_state[self.ctr].append(x)
        self.record_control[self.ctr].append(u)
        self.record_cost[self.ctr].append(cost)

    def plot(self):
        record_estimate = np.array(self.record_estimate)
        record_state = np.array(self.record_state)
        record_control = np.array(self.record_control)
        record_cost = np.array(self.record_cost)

        T_sim = record_state.shape[1]
        time_lin = np.linspace(0, self.dt * T_sim, T_sim)

        fig, (ax1, ax2) = plt.subplots(2, 1)
        for k in range(self.ctr + 1):
            ax1.plot(time_lin, record_control[k, :, 0], label=self.label[k])
            ax2.plot(time_lin, record_control[k, :, 1], label=self.label[k])
        ax1.grid()
        ax2.grid()
        ax1.set_ylabel(r"$u_1$")
        ax2.set_ylabel(r"$u_2$")
        ax1.legend()

        legend = [
            r"$p_x$",
            r"$p_y$",
            r"$\theta$",
            r"$\dot p_x$",
            r"$\dot p_y$",
            r"$\dot \theta$",
            "mass",
        ]
        fig, axs = plt.subplots(7, 1, figsize=(10, 20))

        for k in range(self.ctr + 1):
            for i in range(7):
                axs[i].plot(
                    time_lin, record_state[k, :, i], label=self.label[k] + "_state"
                )
                axs[i].plot(
                    time_lin,
                    record_estimate[k, :, i],
                    "--",
                    label=self.label[k] + "_est",
                )

                if k == 0:
                    axs[i].grid()
                    axs[i].set_ylabel(legend[i])

        axs[0].plot(time_lin, np.array(self.xdes_ref)[:T_sim], ":", label="Reference")
        axs[1].plot(time_lin, np.array(self.ydes_ref)[:T_sim], ":")
        axs[0].legend()

        plt.figure(figsize=(20, 8))
        plt.plot(time_lin, record_state[0, :, -1], color="black", label="Ground truth")
        for k in range(self.ctr + 1):
            plt.plot(
                time_lin,
                record_estimate[k, :, -1],
                "--",
                color=line_styles[k],
                label=self.label[k] + " estimate",
            )

        plt.xlabel(r"Time [s]")
        plt.ylabel(r"m [kg]")
        plt.xlim(0, 4)
        plt.legend()
        plt.tight_layout()

        plt.savefig(FIGURE_PATH + "mass-plot" + ".pdf", bbox_inches="tight")

        plt.figure(figsize=(20, 10))
        for k in range(self.ctr + 1):
            plt.plot(
                time_lin, record_cost[k, :], color=line_styles[k], label=self.label[k]
            )

        plt.xlabel(r"Time [s]")
        plt.ylabel(r"Cost")
        plt.xlim(0, 4)
        plt.legend()
        plt.tight_layout()

        rsekf = record_cost[0]
        print(self.label[0] + " mean cost ", np.mean(rsekf))

        ekf = record_cost[1]
        print(self.label[1] + " mean cost ", np.mean(ekf))

        Gain = (np.mean(rsekf) - np.mean(ekf)) / np.mean(ekf)
        print("Cost gain ", Gain * 100, " %")
        print("\n")

        fig, (ax1, ax2) = plt.subplots(2, 1)
        mmae = []
        for k in range(self.ctr + 1):
            errorx = record_state[k, :, 0] - np.array(self.xdes_ref[:T_sim])
            errory = record_state[k, :, 1] - np.array(self.ydes_ref[:T_sim])

            ax1.plot(
                time_lin,
                np.abs(errorx),
                "--",
                label=self.label[k] + "_error",
            )
            ax2.plot(time_lin, np.abs(errory), "--")

            mse = (errorx ** 2 + errory ** 2) / 2
            mmae.append(np.mean(mse))

            print(self.label[k], " MSE ", mmae[k])

        print("MSE gain = ", ((mmae[0] - mmae[1]) / mmae[1]) * 100, " %")

        ax1.grid()
        ax2.grid()
        ax1.set_ylabel(r"$p_x$")
        ax2.set_ylabel(r"$p_y$")
        ax1.legend()

        plt.figure(figsize=(20, 8))
        for k in range(self.ctr + 1):
            plt.plot(
                record_state[k, :, 0],
                record_state[k, :, 1],
                color=line_styles[k],
                label=self.label[k],
            )
        plt.plot(self.xdes_ref, self.ydes_ref, "-.", color="black", label="reference")
        plt.xlim(0, 1)
        plt.xlabel(r"$p_x$ [m]")
        plt.ylabel(r"$p_z$ [m]")
        plt.legend()

        plt.tight_layout()
        plt.savefig(FIGURE_PATH + "xz-plot" + ".pdf", bbox_inches="tight")

        plt.figure()
        for k in range(self.ctr + 1):
            plt.plot(time_lin, record_cost[k], label=self.label[k])
        plt.legend()
        plt.grid()

        # Compare each filter to the first one

        print(self.label[0] + " Average cost = ", np.mean(record_cost[0]))
        for k in range(self.ctr):
            print("\n")
            print(self.label[k + 1] + " Average cost = ", np.mean(record_cost[k + 1]))
            print(
                "Improvement = ",
                (1 - np.mean(record_cost[k + 1]) / np.mean(record_cost[0])) * 100,
                " %",
            )

        plt.show()
