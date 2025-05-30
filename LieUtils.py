import numpy as np
import cv2
import sys

class LieUtils:
    def __init__(self) -> None:
        self.tolerance = 1e-12
    
    def skew(self,vec:np.array):
        vec = vec.reshape(3)
        S = np.array([
            0, -vec[2], vec[1],
            vec[2], 0, -vec[0],
            -vec[1], vec[0], 0
        ]).reshape(3,3)
        return S.astype(np.float64)

    def vee(self, S):
        s = np.array([-S[1,2],S[0,2],-S[0,1]]).reshape(3,1).astype(np.float64)
        return s
    
    def Exp_SO3(self,vec:np.array):
        vec = vec.reshape(3,1)
        R_mat, _ = cv2.Rodrigues(vec)
        return R_mat
    
    def Log_SO3(self, R : np.array):
        R = R.reshape(3,3)
        psi, _ = cv2.Rodrigues(R)
        psi = psi.reshape(3,1)
        if psi[0,0]<0 and psi[1,0]<0 and psi[2,0]<0 and np.linalg.norm(psi)>(np.pi-0.0001):
            psi = -psi
        return psi

    # def Exp_SO3(self,vec:np.array):
    #     angle = np.linalg.norm(vec)
    #     S = self.skew(vec.copy())
    #     if angle > self.tolerance:
    #         R = np.eye(3) + (np.sin(angle)/angle) * S + (1-np.cos(angle))/(angle**2) * S @ S
    #     else:
    #         R = np.eye(3) + S + 0.5*(S@S)
    #     return R.astype(np.float64)

    # def Log_SO3(self, R : np.array):
    #     R = R.reshape(3,3).astype(np.float64)
    #     gamma = 0.5 * (np.trace(R) - 1).astype(np.float64)
        
    #     if np.abs(gamma) < 1 - self.tolerance:
    #         theta = np.arccos(gamma)
            
    #         if np.abs(theta) > self.tolerance:
    #             alpha = 0.5 * theta / np.sin(theta)
    #             S = R - R.T
    #             psi = alpha * np.array([-S[1,2],S[0,2],-S[0,1]]).reshape(3,1).astype(np.float64)
    #         else:
    #             print("----------------")
    #             print(R)
    #             psi, _ = cv2.Rodrigues(R)
    #             print(psi)
    #             print("----------------")
    #             return psi
            
    #     else:
    #         print("**************")
    #         print(R)
    #         psi, _ = cv2.Rodrigues(R)
    #         print(psi)
    #         print("**************")
    #         return psi

    #     return psi
    
    def J_l_SO3(self, vec:np.array):
        vec = vec.reshape(3,1)
        angle = np.linalg.norm(vec)
        S = self.skew(vec)
        if angle > self.tolerance:
            JL = np.eye(3) + ( (1 - np.cos(angle)) / (angle**2) ) * S + ((angle - np.sin(angle))/(angle**3))*(S@S) 
        else:
            JL = np.eye(3) + (1/2)*S + (1/6)*(S@S) 
        return JL.reshape(3,3).astype(np.float64)
    
    def J_r_SO3(self,vec:np.array):
        return self.J_l_SO3(-vec)
    
    def J_l_inv_SO3(self, vec:np.array):
        vec = vec.reshape(3,1)
        angle = np.linalg.norm(vec)
        S = self.skew(vec)
        if angle > self.tolerance:
            mag1 = -0.5
            mag2 = (1/(angle**2)) - ((1+np.cos(angle))/(2*angle*np.sin(angle)))
            JL_inv = np.eye(3) + mag1*S + mag2*(S@S) 
        else:
            JL_inv = np.eye(3) - (1/2)*S + (1/6)*(1/2)*(S@S) 
        return JL_inv.reshape(3,3).astype(np.float64)

    def J_r_inv_SO3(self, vec:np.array):
        return self.J_l_inv_SO3(vec=(-vec))
    
    def Exp_SE3(self, xi:np.array):
        xi = xi.reshape(6,1)
        psi = xi[0:3,0].reshape(3,1) # First 3 items
        rho = xi[3:6,0].reshape(3,1) # Last  3 items
        
        T = np.eye(4)
        T[:3,:3] = self.Exp_SO3(psi)
        T[:3,3]  = (self.J_l_SO3(psi) @ rho).reshape(3)
        return T
    
    def Log_SE3(self, T:np.array):
        T = T.reshape(4,4)
        R = T[:3, :3] # 3x3 Rotation Matrix
        t = T[:3, 3].reshape(3,1)  # 3x1 Translation Vector
        
        psi = self.Log_SO3(R).reshape(3)
        rho = ( self.J_l_inv_SO3(psi) @ t ).reshape(3)
        
        xi = np.zeros((6,1))
        xi[:3,0] = psi
        xi[3:,0] = rho
        
        return xi
    
    def detarho_detapsi(self, xi:np.array):
        xi = xi.reshape(6,1)
        T = self.Exp_SE3(xi=xi)
        t = T[:3, 3].reshape(3,1)  # 3x1 Translation Vector

        psi = xi[:3,0]

        alpha = np.linalg.norm(psi)
        
        if alpha < self.tolerance:
            print("Not varified operation has been executed")
            return 0.5 * self.skew(t)
        
        A = (1/(alpha**2)) - ((1+np.cos(alpha))/(2*alpha*np.sin(alpha)))
        dA_dalpha = (-2/(alpha**3)) - ( (-np.sin(alpha)*(2*alpha*np.sin(alpha))-(1+np.cos(alpha))*(2*np.sin(alpha)+ 2*alpha*np.cos(alpha)) ) / ((2*alpha*np.sin(alpha))**2) )

        dalpha_dpsi = psi/alpha

        dA_dpsi = dA_dalpha * dalpha_dpsi

        S = self.skew(psi)
        B = (S @ S) @ t.reshape(3,1)

        dB_dpsi = (psi.reshape(1,3) @ t.reshape(3,1))*np.eye(3) - 2 * t.reshape(3,1) @ psi.reshape(1,3) + psi.reshape(3,1) @ t.reshape(1,3)
        dB_dpsi = dB_dpsi.reshape(3,3)

        drho_dpsi = 0.5*self.skew(t) + B.reshape(3,1) @ dA_dpsi.reshape(1,3) + A * dB_dpsi
        return drho_dpsi.reshape(3,3).astype(np.float64)
    
    def J_r_inv_SE3(self,xi:np.array):
        xi = xi.reshape(6,1)
        

        xi_psi = xi[:3,0]
        
        d_etapsi_d_xipsi = self.J_r_inv_SO3(xi_psi)
        d_etapsi_d_xirho = np.zeros((3,3))
        
        d_etarho_d_etapsi = self.detarho_detapsi(xi)
        d_etarho_d_xipsi = d_etarho_d_etapsi @ d_etapsi_d_xipsi
        
        d_etarho_d_xirho = d_etapsi_d_xipsi.copy()
        
        d_eta_d_xi = np.zeros((6,6))
        d_eta_d_xi[:3,:3] = d_etapsi_d_xipsi
        d_eta_d_xi[:3,3:] = d_etapsi_d_xirho
        d_eta_d_xi[3:,:3] = d_etarho_d_xipsi
        d_eta_d_xi[3:,3:] = d_etarho_d_xirho
        
        return d_eta_d_xi.reshape(6,6).astype(np.float64)
    
    def Ad_SE3(self, T:np.array):
        T = T.reshape(4,4)
        R = T[:3,:3].reshape(3,3)
        t = T[:3,3].reshape(3,1)
        
        Ad_T = np.zeros((6,6))
        Ad_T[:3,:3] = R
        Ad_T[3:,3:] = R 
        
        Ad_T[3:,:3] = self.skew(t) @ R
        
        return Ad_T.reshape(6,6).astype(np.float64)
    
    def Exp_SE2_3(self, xi:np.array):
        xi = xi.reshape(9,1)
        psi = xi[0:3,0].reshape(3,1)
        rho = xi[3:6,0].reshape(3,1) 
        nu  = xi[6:9,0].reshape(3,1) 
        
        T = np.eye(5)
        T[:3,:3] = self.Exp_SO3(psi)
        T[:3,3]  = (self.J_l_SO3(psi) @ rho).reshape(3)
        T[:3,4]  = (self.J_l_SO3(psi) @ nu).reshape(3)
        return T
    
    def Log_SE2_3(self, T:np.array):
        T = T.reshape(5,5)
        R = T[:3, :3]               # 3x3 Rotation Matrix
        t = T[:3, 3].reshape(3,1)   # 3x1 Translation Vector
        v = T[:3, 4].reshape(3,1)   # 3x1 Velocity Vector
        
        psi = self.Log_SO3(R).reshape(3)
        rho = ( self.J_l_inv_SO3(psi) @ t ).reshape(3)
        nu  = ( self.J_l_inv_SO3(psi) @ v ).reshape(3)
        
        xi = np.zeros((9,1))
        xi[0:3,0] = psi
        xi[3:6,0] = rho
        xi[6:9,0] = nu
        
        return xi

    def Ad_SE2_3(self, T:np.array):
        T = T.reshape(5,5)
        
        R = T[:3,:3].reshape(3,3)
        t = T[:3,3].reshape(3,1)
        v = T[:3,4].reshape(3,1)
        
        Ad_T = np.zeros((9,9))
        for idx in range(3):
            curr_id = 3 * idx
            next_id = 3 * (idx + 1)
            Ad_T[curr_id:next_id,curr_id:next_id] = R

        Ad_T[3:6,0:3] = self.skew(t) @ R
        Ad_T[6:9,0:3] = self.skew(v) @ R
        
        return Ad_T.reshape(9,9).astype(np.float64)
        