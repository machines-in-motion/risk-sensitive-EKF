
import numpy as np
import enum
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as m_patches
from matplotlib.collections import PatchCollection
import pickle
import sys

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
line_styles = 10*['g', 'b', 'r', 'c', 'y', 'k', 'm']



class DataRecorder:

    def __init__(self):
        self.dt = None
        self.estimator_type = None
        self.pose_ref = None
        self.record_estimate = []
        self.record_COM_mes = []
        self.record_push = []
        self.record_f = []
        self.record_f_py = []
        self.record_q = []
        self.record_v = []
        self.record_vdes = []
        self.record_tau = []

        self.ctr = -1

    def add_experiment(self, n_step):
    
        self.ctr += 1

        self.record_estimate.append([np.zeros(15)]*n_step)
        self.record_COM_mes.append([np.zeros(9)]*n_step)
        self.record_push.append([None]*n_step)
        self.record_f.append([np.zeros(12)]*n_step)
        self.record_f_py.append([None]*n_step)
        self.record_q.append([np.zeros(19)]*n_step)
        self.record_v.append([np.zeros(18)]*n_step)
        self.record_vdes.append([np.zeros(18)]*n_step)
        self.record_tau.append([np.zeros(12)]*n_step)

    def record_data(self, step, x_init, x_COM_mes, f, q, v, v_des, tau, ft_obj=None, f_GT=None):

        self.record_estimate[self.ctr][step] = x_init
        self.record_COM_mes[self.ctr][step] = x_COM_mes
        self.record_push[self.ctr][step] = ft_obj
        self.record_f[self.ctr][step] = f
        self.record_f_py[self.ctr][step] = f_GT
        self.record_q[self.ctr][step] = q
        self.record_v[self.ctr][step] = v
        self.record_vdes[self.ctr][step] = v_des
        self.record_tau[self.ctr][step] = tau


    def smooth(self, signal):
        avg = [np.mean(signal[i:i+10]) for i in range(len(signal))]
        return np.array(avg)

    def plot(self):
        record_estimate = np.array(self.record_estimate)
        record_COM_mes = np.array(self.record_COM_mes)
        record_push = np.array(self.record_push)
        record_f = np.array(self.record_f)
        record_f_py = np.array(self.record_f_py)
        record_q = np.array(self.record_q)
        record_v = np.array(self.record_v)
        record_vdes = np.array(self.record_vdes)
        record_tau = np.array(self.record_tau)


        T_sim = record_COM_mes.shape[1]
        time_lin = np.linspace(0, self.dt * T_sim, T_sim)

        fig, axs = plt.subplots(3, 1)
        for k in range(self.ctr+1):
            for i in range(3):
                axs[i].plot(time_lin, record_v[k][:, 6+i], label=self.estimator_type[k]+ "_v")
                axs[i].plot(time_lin, record_vdes[k][:, 6+i], "-.", label=self.estimator_type[k]+ "_v_des")
                if k == 0:
                    axs[i].grid()
                axs[i].set_ylabel("Joint " + str(i+1))
            axs[0].legend()


        fig, axs = plt.subplots(9, 1)
        labels = [r"$p_x$", r"$p_y$", r"$p_z$", r"$v_x$", r"$v_y$", r"$v_z$", r"$L_x$", r"$L_y$", r"$L_z$"]
        for k in range(self.ctr+1):
            for i in range(9):
                axs[i].plot(time_lin, record_COM_mes[k][:, i], label=self.estimator_type[k]+ "_mes")
                axs[i].plot(time_lin, record_estimate[k][:, i], "-.", label=self.estimator_type[k]+ "_Estimate")
                if k == 0:
                    axs[i].grid()
                axs[i].set_ylabel(labels[i])
            axs[0].legend()

        plt.figure(figsize=(20, 10))
        for k in range(self.ctr+1):
            esti = self.smooth(list(record_estimate[k][:, 2]))
            plt.plot(time_lin, record_COM_mes[k][:, 2], color=line_styles[k], label=self.estimator_type[k] + " measurement")
            plt.plot(time_lin, esti, "--", color=line_styles[k], label=self.estimator_type[k] + " estimate")

        plt.plot(time_lin, np.array([0.]*len(time_lin)), "--", color="black") #, label="Target")
        plt.ylabel(r"$p_z$ [m]")
        plt.xlabel(r"Time [s]")
        plt.xlim(0., 25.)
        # plt.ylim(0.08, 0.3)
        plt.legend()
        plt.savefig(FIGURE_PATH+"zplot"+".pdf", bbox_inches='tight')


        t_cut = 0
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12.2))
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(40, 15))
        for k in range(self.ctr+1):
            esti = self.smooth(list(record_estimate[k][t_cut:, 2]))
            ax1.plot(time_lin[:], record_COM_mes[k][t_cut:, 2], color=line_styles[k], label=self.estimator_type[k] + " measurement")
            ax1.plot(time_lin[:], esti, "-.", color=line_styles[k], label=self.estimator_type[k] + " estimate")

            esti = self.smooth(list(record_estimate[k][t_cut:, 9+2]))
            ax2.plot(time_lin[:], esti, "-.", color=line_styles[k], label=self.estimator_type[k]+ "_estimate")
        ax1.plot(time_lin, np.array([0.]*len(time_lin)), "--", color="black") #, label="Target")
        # ax1.axvline(x = 1.4, color = 'black')
        # ax2.axvline(x = 1.4, color = 'black')
        ax1.set_xticklabels([]) 
        ax1.set_ylabel(r"$p_z$ [m]")
        ax2.set_ylabel(r"$F_{ext}^z [N]$")
        ax2.set_xlabel(r"Time [s]")
        ax1.set_xlim(0., 6.)
        ax2.set_xlim(0., 6.)
        ax1.legend()
        plt.tight_layout()
        plt.savefig(FIGURE_PATH+"zf-plot"+".pdf", bbox_inches='tight')


        fig, axs = plt.subplots(6, 1)
        labels = [r"$df_x$", r"$df_y$", r"$df_z$", r"$d\tau_x$", r"$d\tau_y$", r"$d\tau_z$"]
        for k in range(self.ctr+1):
            for i in range(6):
                if(not None in record_push):
                    axs[i].plot(time_lin, record_push[k][:, i], "-.", label=self.estimator_type[k]+ "_mes")
                axs[i].plot(time_lin, record_estimate[k][:, 9 + i], "-.", label=self.estimator_type[k]+ "_Estimate")
                axs[i].set_xlim(0., 25.)

                axs[i].set_ylabel(labels[i])
                axs[i].axvline(x = 6.4, color = 'black')
                if i != 5:
                    axs[i].set_xticklabels([])
            axs[0].legend(loc="upper right")

        labels = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
        fig, axs = plt.subplots(4, 3, constrained_layout=True)
        t_cut = 100
        for k in range(self.ctr+1):
            for i in range(4):
                axs[i, 0].plot(time_lin, record_f[k][:, i * 3 + 0], label=self.estimator_type[k])
                axs[i, 1].plot(time_lin, record_f[k][:, i * 3 + 1], label=self.estimator_type[k])
                axs[i, 2].plot(time_lin, record_f[k][:, i * 3 + 2], label=self.estimator_type[k])
                if(not None in record_f_py):
                    axs[i, 0].plot(time_lin[t_cut:], record_f_py[k][t_cut:, i, 0], "-.", label=self.estimator_type[k]+ "_py")
                    axs[i, 1].plot(time_lin[t_cut:], record_f_py[k][t_cut:, i, 1], "-.", label=self.estimator_type[k]+ "_py")
                    axs[i, 2].plot(time_lin[t_cut:], record_f_py[k][t_cut:, i, 2], "-.", label=self.estimator_type[k]+ "_py")
                if k == 0:
                    axs[i, 0].grid()
                    axs[i, 1].grid()
                    axs[i, 2].grid()
                    axs[i, 2].set_ylim(0, 10)
                axs[i, 0].set_ylabel(labels[i])
            axs[0, 0].legend()
            axs[0, 1].legend()
            axs[0, 2].legend()

            axs[3, 0].set_xlabel(r"F_x")
            axs[3, 1].set_xlabel(r"F_y")
            axs[3, 2].set_xlabel(r"F_z")


        fig, axs = plt.subplots(12, 1)
        for k in range(self.ctr+1):
            for i in range(12):
                axs[i].plot(time_lin, record_tau[k][:, i], label=self.estimator_type[k]+ "")
                if k == 0:
                    axs[i].grid()
                axs[i].set_ylabel(r"$\tau$_" + str(i+1))
            axs[0].legend()

        plt.show()


    def save(self, file_name):

        l =  [self.record_estimate, self.record_COM_mes, self.record_push, self.record_f, self.record_f_py, self.record_q, self.record_v, self.record_vdes, self.record_tau, self.estimator_type[k], self.dt, self.pose_ref]

        with open("data/" + file_name, "wb") as fp:
            pickle.dump(l, fp)



    def load(self, file_name):
        with open("data/" + file_name, "rb") as fp:
            b = pickle.load(fp)
            
        self.record_estimate = b[0]
        self.record_COM_mes = b[1]
        self.record_push = b[2]
        self.record_f = b[3]
        self.record_f_py = b[4]
        self.record_q = b[5]
        self.record_v = b[6]
        self.record_vdes = b[7]
        self.record_tau = b[8]
        self.estimator_type = [b[9]]
        self.dt = b[10]
        # self.pose_ref = b[11]
        self.ctr = 0

    def add_load(self, file_name, shift=0):
        with open("data/" + file_name, "rb") as fp:
            b = pickle.load(fp)

        self.record_estimate += b[0]
        self.record_COM_mes += b[1]
        self.record_push += b[2]
        self.record_f += b[3]
        self.record_f_py += b[4]
        self.record_q += b[5]
        self.record_v += b[6]
        self.record_vdes += b[7]
        self.record_tau += b[8]
        self.estimator_type += [b[9]]

        if shift > 0:
            self.record_estimate[-1][:-shift] = self.record_estimate[-1][shift:]
            self.record_estimate[-1][-shift:] = [np.zeros(15)]*shift  

            self.record_COM_mes[-1][:-shift] = self.record_COM_mes[-1][shift:]
            self.record_COM_mes[-1][-shift:] = [np.zeros(9)]*shift  

            # self.record_push[-1][:-shift] = self.record_push[-1][shift:]
            # self.record_push[-1][-shift:] = [np.zeros(15)]*shift  

            self.record_q[-1][:-shift] = self.record_q[-1][shift:]
            self.record_q[-1][-shift:] = [np.zeros(19)]*shift  

            self.record_v[-1][:-shift] = self.record_v[-1][shift:]
            self.record_v[-1][-shift:] = [np.zeros(18)]*shift  

            self.record_vdes[-1][:-shift] = self.record_vdes[-1][shift:]
            self.record_vdes[-1][-shift:] = [np.zeros(18)]*shift  

            self.record_tau[-1][:-shift] = self.record_tau[-1][shift:]
            self.record_tau[-1][-shift:] = [np.zeros(12)]*shift  



        self.estimator_type = ["RS-EKF", "EKF"]


        assert self.dt == b[10], "Both data should have the same dt."
        self.ctr += 1

if __name__ == "__main__":


    try:
        file_name1 = sys.argv[1]
    except:
        file_name1 = "test"
    

    try:
        file_name2 = sys.argv[2]
    except:
        file_name2 = None



    print(f"File name 1: {file_name1}")


    dr = DataRecorder()
    dr.load(file_name1)


    if file_name2 is not None:
        print(f"File name 2: {file_name2}")
        # dr.add_load(file_name2,3650)
        dr.add_load(file_name2)



    # dr.estimator_type = ["RS-EKF", "EKF"]
    dr.plot()