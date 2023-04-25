
import numpy as np
from mim_data_utils import DataLogger, DataReader

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as m_patches
from matplotlib.collections import PatchCollection

import sys

try:
    file_name = sys.argv[1]
except:
    file_name = "test"

print(f"File name : {file_name}")


import subprocess
bashCommand = "sudo chmod -R a+w ./data"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

names = ["q", "v","x_ref", "r", "x_init", "f", "x_GT", "q_start"]
reader = DataReader("data/" + file_name +".mds")



labels = ["FL", "FR", "HL", "HR"]
fig, axs = plt.subplots(4, 3, constrained_layout=True)
for i in range(4):
    axs[i, 0].plot(reader.data[names[5]][:, 3*i], label="Fx")
    axs[i, 1].plot(reader.data[names[5]][:, 3*i+1], label="Fy")
    axs[i, 2].plot(reader.data[names[5]][:, 3*i+2], label="Fz")
    axs[i, 0].grid()
    axs[i, 1].grid()
    axs[i, 2].grid()
    axs[i, 0].set_ylabel(labels[i])
axs[0, 0].legend()
axs[0, 1].legend()
axs[0, 2].legend()

axs[3, 0].set_xlabel(r"$F_x$")
axs[3, 1].set_xlabel(r"$F_y$")
axs[3, 2].set_xlabel(r"$F_z$")
fig.suptitle('Force', fontsize=16)



labels = [r"$p_x$", r"$p_y$", r"$p_z$", r"$v_x$", r"$v_y$", r"$v_z$", r"$L_x$", r"$L_y$", r"$L_z$"]
fig, axs = plt.subplots(9, 1, constrained_layout=True)
for i in range(9):
    axs[i].plot(reader.data[names[6]][:, i], label="GT")
    axs[i].plot(reader.data[names[4]][:, i], "--", label="EST")
    axs[i].plot(reader.data[names[2]][:, i], label="REF")
    axs[i].set_ylabel(labels[i])
    axs[i].grid()
axs[0].legend()
fig.suptitle('COM state', fontsize=16)



labels = [r"$df_x$", r"$df_y$", r"$df_z$", r"$d\tau_x$", r"$d\tau_y$", r"$d\tau_z$"]
fig, axs = plt.subplots(6, 1, constrained_layout=True)
for i in range(6):
    axs[i].plot(reader.data[names[4]][:, 9+i], label="EST")
    axs[i].grid()
    axs[i].set_ylabel(labels[i])
axs[0].legend()
fig.suptitle('Estimated Force', fontsize=16)

from scipy.spatial.transform import Rotation as R



labels = [r"$base_x$", r"$base_y$", r"$base_z$", r"$orientation_1$", r"$orientation_2$", r"$orientation_3$",  r"$orientation_4$"]
fig, axs = plt.subplots(7, 1, constrained_layout=True)
for i in range(7):
    axs[i].plot(reader.data[names[0]][:, i], label="measurement")
    axs[i].grid()
    axs[i].set_ylabel(labels[i])
axs[0].legend()
fig.suptitle('Joint position', fontsize=16)

quaternions = reader.data[names[0]][:, 3:7]
euler = []
for i in range(len(quaternions)):
    r = R.from_quat(quaternions[i])
    euler.append(r.as_euler('xyz', degrees=True))
euler = np.array(euler)


# to do: check euler convention
labels = [r"$Euler_x$", r"$Euler_y$", r"$Euler_z$"]
fig, axs = plt.subplots(3, 1, constrained_layout=True)
for i in range(3):
    axs[i].plot(euler[:, i], label="measurement")
    axs[i].grid()
    axs[i].set_ylabel(labels[i])
axs[0].legend()
fig.suptitle('Euler angles', fontsize=16)


plt.show()