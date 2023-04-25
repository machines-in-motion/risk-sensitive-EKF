import numpy as np
import pybullet as p
import pinocchio as pin
import proxsuite


class Disturbances:
    def __init__(self, robotid, estimator, T_sim, time_seq, forces, torques):
        self.estimator = estimator
        self.robotid = robotid

        time_seq += [T_sim]

        self.forceObj = [np.zeros(3)] * time_seq[0]
        self.torqueObj = [np.zeros(3)] * time_seq[0]
        for i in range(len(time_seq)-1):
            T = time_seq[i+1] - time_seq[i]
            self.forceObj += [np.array(forces[i])] * T
            self.torqueObj += [np.array(torques[i])] * T

    def applyExternalDisturbance(self, t, q, v):
        posObj = self.estimator.get_centroidal_state(q, v)[:3]
        p.applyExternalForce(self.robotid, -1, self.forceObj[t], posObj, p.WORLD_FRAME)
        p.applyExternalTorque(self.robotid, -1, self.torqueObj[t], p.WORLD_FRAME)
        return  np.concatenate([self.forceObj[t], self.torqueObj[t]])


class Block:
    def __init__(self, robotid, filter_type, RECORD):
        self.robotid = robotid
        self.cubeStartPos = np.array([0.15,0.,0.3])     
        p.resetDebugVisualizerCamera(
            cameraDistance=0.6,
            cameraYaw=0,
            cameraPitch=-25,
            cameraTargetPosition=[0.2, 0, 0],
        )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        if RECORD:
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "./" + filter_type + ".mp4")
        

    def applyExternalDisturbance(self, t, q, v):
        if t==2000:
            block = p.loadURDF("block.urdf", self.cubeStartPos)
        return None




def get_contact(robotid, endeff_if):
    n_eff = len(endeff_if)
    f_GT = np.zeros((n_eff, 3))


    endeff_if = [x-1 for x in endeff_if]  

    contact_info = list(p.getContactPoints(robotid))
    active_contact = [x[3] for x in contact_info]

    for i in range(n_eff):
        mask = (np.array(active_contact) == np.array([endeff_if[i]]*len(active_contact)))

        if np.sum(mask) != 0:
            contactNormalOnB = np.array([x[7] for x in contact_info])[mask]
            normal_force = np.array([x[9] for x in contact_info])[mask]
            lateralFriction1 = np.array([x[10] for x in contact_info])[mask]
            lateralFrictionDir1 = np.array([x[11] for x in contact_info])[mask]
            lateralFriction2 = np.array([x[12] for x in contact_info])[mask]
            lateralFrictionDir2 = np.array([x[13] for x in contact_info])[mask]
            n = np.sum(mask)
            f_GT[i] += np.sum(np.array([normal_force[i] * contactNormalOnB[i]for i in range(n)]), axis=0)
            f_GT[i] += np.sum(np.array([lateralFriction1[i] * lateralFrictionDir1[i]for i in range(n)]), axis=0)
            f_GT[i] += np.sum(np.array([lateralFriction2[i] * lateralFrictionDir2[i]for i in range(n)]), axis=0)
    
    return f_GT


class Inverse_Dynamics():

    def __init__(self, pin_robot, dt_ctrl):
        self.pin_robot = pin_robot
        self.model = pin_robot.model
        self.data = pin_robot.data
        self.hl_index = self.pin_robot.model.getFrameId("HL_ANKLE")
        self.hr_index = self.pin_robot.model.getFrameId("HR_ANKLE")
        self.fl_index = self.pin_robot.model.getFrameId("FL_ANKLE")
        self.fr_index = self.pin_robot.model.getFrameId("FR_ANKLE")
        self.dt_ctrl = dt_ctrl

        self.S = np.zeros((18, 12))
        self.S[6:] = np.eye(12)


        self.H = np.zeros((42, 42))
        self.H[18:30, 18:30] = np.eye(12)


        # self.C = np.zeros((36, 42))
        # self.C[:, 6:] = np.eye(36)
        self.C = np.zeros((24, 42))
        self.C[:, 18:] = np.eye(24)


        self.u2 = np.array([4, 4, 20] * 4 + [2]*12)
        self.l2 = np.array([-4, -4, 0] * 4 + [-2]*12)


        self.q_min = np.array([-0.7, 0.4, -2.8]*2 + [-0.7, -1.3, 0.6]*2)
        self.q_max = np.array([0.7, 1.3, -0.6]*2 + [0.7, -0.4, 2.8]*2 )


    def get_torque(self, q, v, F_des, noFriction=True):

        pin.updateFramePlacements(self.model, self.data)
        pin.computeCentroidalMap(self.model, self.data, q)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, v)
        M = pin.crba(self.model, self.data, q)

        # compute dynamic drift -- Coriolis, centrifugal, gravity
        h = pin.rnea(self.model, self.data, q, v, np.zeros(self.model.nv))

        J1 = pin.computeFrameJacobian(self.model, self.data, q, self.fl_index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]
        J2 = pin.computeFrameJacobian(self.model, self.data, q, self.fr_index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]
        J3 = pin.computeFrameJacobian(self.model, self.data, q, self.hl_index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]
        J4 = pin.computeFrameJacobian(self.model, self.data, q, self.hr_index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]

        dJ1 = pin.getFrameJacobianTimeVariation(self.model, self.data, self.fl_index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]
        dJ2 = pin.getFrameJacobianTimeVariation(self.model, self.data, self.fr_index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]
        dJ3 = pin.getFrameJacobianTimeVariation(self.model, self.data, self.hl_index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]
        dJ4 = pin.getFrameJacobianTimeVariation(self.model, self.data, self.hr_index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]

        Jc = np.concatenate((J1, J2, J3, J4))
        dJc = np.concatenate((dJ1, dJ2, dJ3, dJ4))

        g = np.zeros(42)
        g[18:30] = - F_des

        A1 = np.concatenate((M, -Jc.T, -self.S), axis=1)
        A2 =  np.concatenate((Jc, np.zeros((12, 24))), axis=1)

        A = np.concatenate((A1, A2))
        
        if(noFriction):
            friction =  np.zeros(18)
        else:
            friction =  self.S @ (-0.07 * np.arctan(v[6:]/0.5) * 2 / np.pi)

        b = np.concatenate((- h  + friction, -dJc @ v))


        # l1 = (self.q_min - q[7:] - self.dt_ctrl * v[6:]) * 2 / self.dt_ctrl ** 2
        # u1 = (self.q_max - q[7:] - self.dt_ctrl * v[6:]) * 2 / self.dt_ctrl ** 2
        # u = np.concatenate((u1, self.u2))
        # l = np.concatenate((l1, self.l2))

        qp = proxsuite.proxqp.dense.QP(42, 30, 24)  #  n, n_eq, n_in
        qp.init(self.H, g, A, b, self.C, self.l2, self.u2)
        # qp = proxsuite.proxqp.dense.QP(42, 30, 36)  #  n, n_eq, n_in
        # qp.init(self.H, g, A, b, self.C, l, u)
        qp.solve()

        # print("N_it ", qp.results.info.iter)
        # if qp.results.info.iter > 10000:
        #     import pickle
        #     l = [self.H, g, A, b, self.C, self.l, self.u]
        #     with open("data/" + "hardQP", "wb") as fp:   #Pickling
        #         pickle.dump(l, fp)


        a = qp.results.x[:18]
        F = qp.results.x[18:30]
        tau = qp.results.x[30:]
        return a, F, tau