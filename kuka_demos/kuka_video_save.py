import os, sys
src_path = os.path.abspath("../")
sys.path.append(src_path)
from kuka_model.kuka_circle import KukaCircle, measurement_model_full, measurement_model, get_p_
from bullet_utils.env import BulletEnvWithGround
from robot_properties_kuka.iiwaWrapper import IiwaRobot
import numpy as np 
import pybullet as p
import time

dt_sim = 1e-3
env = BulletEnvWithGround(p.GUI, dt=dt_sim)
robot_simulator = IiwaRobot()
horizon = 10
dt = 0.05
kukacircle = KukaCircle(robot_simulator, horizon, dt=dt, dt_sim=dt_sim)



xtraj_rs = np.load( "kuka_push_RS-EKF.npy")
xtraj = np.load( "kuka_push_EKF.npy")


circle = list(kukacircle.target)
color = [[0, 0, 0]] * len(kukacircle.target)
p.addUserDebugPoints(circle, color, pointSize=3)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

p.resetDebugVisualizerCamera(
     cameraDistance=1.15,
     cameraYaw=0,
     cameraPitch=-30,
     cameraTargetPosition=[0.35, 0, 0.15],
 )

endeff_frame_id = robot_simulator.pin_robot.model.getFrameId("contact")
trailDuration = 1000000


video_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "./" + "EKF" + ".mp4")

prev_pose = get_p_(xtraj[0, :7], robot_simulator.pin_robot.model, endeff_frame_id)


factor = 8
N = int(len(xtraj) / factor)
for i in range(N): 
    q = xtraj[i*factor, :7]
    v = xtraj[i*factor, 7:]
    robot_simulator.reset_state(q, v)
    
    current_pose = get_p_(q, robot_simulator.pin_robot.model, endeff_frame_id)
    p.addUserDebugLine(prev_pose, current_pose, lineColorRGB=[0, 0, 1], lineWidth=3, lifeTime=trailDuration)
    prev_pose = current_pose


p.stopStateLogging(video_id)

prev_pose = get_p_(xtraj_rs[0, :7], robot_simulator.pin_robot.model, endeff_frame_id)

video_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "./" + "RS_EKF" + ".mp4")

for i in range(N): 
    q = xtraj_rs[i*factor, :7]
    v = xtraj_rs[i*factor, 7:]
    robot_simulator.reset_state(q, v)
    
    current_pose = get_p_(q, robot_simulator.pin_robot.model, endeff_frame_id)
    p.addUserDebugLine(prev_pose, current_pose, lineColorRGB=[0, 0.5, 0], lineWidth=3, lifeTime=trailDuration)
    prev_pose = current_pose

    
p.stopStateLogging(video_id)
