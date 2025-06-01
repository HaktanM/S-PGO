import numpy as np
from StereoSetup import StereoSetup
from utils import SamplePoses


class Manager:
    def __init__(self):
        self.cam = StereoSetup()

        # Get the poses of the cameras
        pose_sampler  = SamplePoses()
        pose_sampler.create_samples_deterministic()
        self.poses    = pose_sampler.poses

        self.anchor_idx = 0
        self.projec_idx = 1

        self.pa_hom = np.array([
            320.0, 100.0, 1.0
        ]).reshape(3,1)
        # self.pa_hom = np.array([
        #     np.random.uniform(0, 640),
        #     np.random.uniform(0, 512),
        #     1.0
        # ]).reshape(3,1)

        self.actual_depth     = 5.0 # np.random.uniform(1.0, 10.0)
        self.actual_inv_depth = 1 / self.actual_depth

        self.estim_depth     = 15 # np.random.uniform(1.0, 10.0)
        self.estim_inv_depth = 1 / self.estim_depth

        self.compute_observation()

        print(f"Actual Depth: {self.actual_depth:.2f}, Estimated Depth: {self.estim_depth:.2f}")

    def step(self):
        observation = self.pn_hom
        estimation  = self.compute_estimated_observation()

        print("observation:\n",observation)
        print("estimation:\n",estimation)

        r    = (observation - estimation)[:2,0].reshape(-1,1)
        J    = self.compute_analytical_Jacobian()[:2,0].reshape(-1,1)

        print("J:\n",J)
        print("r:\n",r)

        H = J.T @ J
        g = J.T @ r

        H += np.eye(H.shape[0]) * 1e-3 # Required for stability

        # # H += np.eye(H.shape[0]) * 1e-6 # Required for stability
        delta_alpha = np.linalg.solve(H, g)

        self.estim_inv_depth += delta_alpha.item()
        self.estim_depth = 1.0 / self.estim_inv_depth

        print(f"Actual Depth: {self.actual_depth:.2f}, Estimated Depth: {self.estim_depth:.2f}")

    def compute_observation(self):
        T_ca_g = self.poses[self.anchor_idx]
        T_cn_g = self.poses[self.projec_idx]

        T_ca_cn = np.linalg.inv(T_cn_g) @ T_ca_g
        R_ca_cn = T_ca_cn[:3,:3]
        t_ca_cn = T_ca_cn[:3,3]

        t_feat_ca = self.cam.Kl_inv @ self.pa_hom / self.actual_inv_depth
        t_feat_cn = (R_ca_cn @ t_feat_ca).reshape(3,1) + t_ca_cn.reshape(3,1)

        pn_hom = self.cam.Kl @ t_feat_cn

        print(f"Observed Depth: {pn_hom[2,0]}")
        pn_hom = pn_hom / pn_hom[2,0]

        self.pn_hom = pn_hom
    
    def compute_estimated_observation(self):
        T_ca_g = self.poses[self.anchor_idx]
        T_cn_g = self.poses[self.projec_idx]

        T_ca_cn = np.linalg.inv(T_cn_g) @ T_ca_g
        R_ca_cn = T_ca_cn[:3,:3]
        t_ca_cn = T_ca_cn[:3,3]

        t_feat_ca = self.cam.Kl_inv @ self.pa_hom / self.estim_inv_depth
        t_feat_cn = (R_ca_cn @ t_feat_ca).reshape(3,1) + t_ca_cn.reshape(3,1)

        pn_hom = self.cam.Kl @ t_feat_cn
        pn_hom = pn_hom / pn_hom[2,0]

        return pn_hom

    def compute_analytical_Jacobian(self):
        T_ca_g = self.poses[self.anchor_idx]
        T_cn_g = self.poses[self.projec_idx]

        T_ca_cn = np.linalg.inv(T_cn_g) @ T_ca_g
        R_ca_cn = T_ca_cn[:3,:3]
        t_ca_cn = T_ca_cn[:3,3]

        t_feat_ca = self.cam.Kl_inv @ self.pa_hom / self.estim_inv_depth
        t_feat_cn = (R_ca_cn @ t_feat_ca).reshape(3,1) + t_ca_cn.reshape(3,1)

        # pn_hom = self.cam.Kl @ t_feat_cn
        # pn_hom = pn_hom / pn_hom[2,0]

        del_pn_del_bn    = self.cam.Kl
        del_bn_del_tn    = self.jacobian_of_projection(vec=t_feat_cn)
        del_tn_del_ta    = R_ca_cn
        del_ta_del_alpha = - t_feat_ca / self.estim_inv_depth

        del_pn_del_alpha = del_pn_del_bn @ del_bn_del_tn @ del_tn_del_ta @ del_ta_del_alpha

        return del_pn_del_alpha

    def compute_numeric_Jacobian(self):
        T_ca_g = self.poses[self.anchor_idx]
        T_cn_g = self.poses[self.projec_idx]

        T_ca_cn = np.linalg.inv(T_cn_g) @ T_ca_g
        R_ca_cn = T_ca_cn[:3,:3]
        t_ca_cn = T_ca_cn[:3,3]

        pert_mag = 1e-4


        # First, the positive perturbation
        t_feat_ca_pp = self.cam.Kl_inv @ self.pa_hom / (self.estim_inv_depth + pert_mag)
        t_feat_cn_pp = (R_ca_cn @ t_feat_ca_pp).reshape(3,1) + t_ca_cn.reshape(3,1)

        pn_hom_pp = self.cam.Kl @ t_feat_cn_pp
        pn_hom_pp = pn_hom_pp / pn_hom_pp[2,0]


        # Seoond, the negative perturbation
        t_feat_ca_nn = self.cam.Kl_inv @ self.pa_hom / (self.estim_inv_depth - pert_mag)
        t_feat_cn_nn = (R_ca_cn @ t_feat_ca_nn).reshape(3,1) + t_ca_cn.reshape(3,1)

        pn_hom_nn = self.cam.Kl @ t_feat_cn_nn
        pn_hom_nn = pn_hom_nn / pn_hom_nn[2,0]

        # Then, compute the numeric derivative
        del_pn_del_alpha = (pn_hom_pp.reshape(3,1) - pn_hom_nn.reshape(3,1)) / (2*pert_mag)

        return del_pn_del_alpha


    def jacobian_of_projection(self, vec:np.array):
        vec = vec.reshape(3)
        result = np.array([
            1.0/vec[2], 0.0, -(vec[0]/(vec[2]*vec[2])),
            0.0, 1.0/vec[2], -(vec[1]/(vec[2]*vec[2])),
            0.0,      0.0,                 0.0
        ]).reshape(3,3)
        return result


if __name__ == "__main__":
    manager = Manager()

    analytical_jacobian = manager.compute_analytical_Jacobian()
    numeric_jacobian    = manager.compute_numeric_Jacobian()

    for idx in range(10):
        manager.step()
