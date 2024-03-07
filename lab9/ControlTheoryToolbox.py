import matplotlib.pyplot as plt
from scipy.linalg import expm
import scipy.integrate as integrate
from scipy.linalg import null_space
from sympy import Matrix, simplify, exp
from sympy import Symbol 
import numpy as np
import control 
import sympy
import os


class System():

    def __init__(self, A, B, C, D):
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        self.D = np.array(D)

        if not self.check_system(self):
            raise Exception("Cannot create system like that")
            

    def check_system(self):
        return (len(self.A.shape) == 2) and (self.A.shape[0] == self.A.shape[1]) \
            and (self.B.shape[0] == self.A.shape[0]) and (self.C.shape[1] == self.A.shape[0]) \
            and (self.D.shape[0] == self.B.shape[0])


class CToolbox():

    def get_jordan_cells(A):
        assert len(A.shape) == 2
        assert A.shape[0] == A.shape[1]
        
        n = A.shape[0]
        jordan_cells_poses = []
        jordan_cells = []
        current_jordan_cell_size = 1
        current_jordan_cell_pos = 0

        while current_jordan_cell_pos + current_jordan_cell_size < n:
            
            jordan_cell_down_element_pos = current_jordan_cell_pos + current_jordan_cell_size - 1
            if A[jordan_cell_down_element_pos, jordan_cell_down_element_pos + 1] == 1 or A[jordan_cell_down_element_pos + 1, jordan_cell_down_element_pos] == 1:
                current_jordan_cell_size += 1
            else:
                jordan_cells_poses.append((current_jordan_cell_pos, current_jordan_cell_size))
                current_jordan_cell_pos += current_jordan_cell_size
                current_jordan_cell_size = 1

        jordan_cells_poses.append((current_jordan_cell_pos, current_jordan_cell_size))
        
        for pos in jordan_cells_poses:
            jordan_cells.append(A[pos[0]:pos[0]+pos[1], pos[0]:pos[0]+pos[1]])

        return jordan_cells_poses, jordan_cells
    
    def get_control_matrix(A, B):
        assert len(A.shape) == 2
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] == B.shape[0]

        n = A.shape[0]
        U = np.zeros((n, 0))
        for i in range(n):
            new_column = (np.linalg.matrix_power(A, i) @ B).reshape((-1, 1))
            U = np.concatenate((U, new_column), axis=1)

        return U
    
    def check_eigenvalues_controllable(A, B, method = 'rank_criteria') -> np.array:
        assert len(A.shape) == 2
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] == B.shape[0]
        
        n = A.shape[0]
        is_controllable = []

        if method == 'rank_criteria':
            eigen_values, _ = np.linalg.eig(A)
            for eigen_value in eigen_values:
                M = np.concatenate((A - eigen_value * np.eye(n), B.reshape(-1, 1)), axis=1)
                is_controllable.append(np.linalg.matrix_rank(M) == n)
        elif method == 'jordan_form':
            P, J = Matrix(A).jordan_form()
            jordan_cells_poses, _ = CToolbox.get_jordan_cells(J)
            B_jordan_form = np.array(P.inv() @ B).T.flatten()
            print('jordan_form ', B_jordan_form)
            for jordan_cells_pose in jordan_cells_poses:
                down_stroke_id = jordan_cells_pose[0] + jordan_cells_pose[1] - 1
                is_controllable.append(True if B_jordan_form[down_stroke_id] != 0 else False) 
        else:
            raise NotImplementedError

        return np.array(is_controllable)
    
    def check_system_controllable(A, B, method = 'rank_criteria'):
        assert len(A.shape) == 2
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] == B.shape[0]
        
        is_controllable = None
        n = A.shape[0]
        if method == 'rank_criteria':
            U = CToolbox.get_control_matrix(A, B)
            is_controllable = np.linalg.matrix_rank(U) == n
        elif method == 'eigen_values_criteria':
            is_controllable = True
            _, J = Matrix(A).jordan_form()
            _, jordan_cells = CToolbox.get_jordan_cells(J)
        
            for i in range(len(jordan_cells)):
                for j in range(len(jordan_cells)):
                    if i != j and np.array_equal(jordan_cells[i], jordan_cells[j]):
                        is_controllable = False
            
            is_controllable = is_controllable and np.all(CToolbox.check_eigenvalues_controllable(A, B))
        else:
            raise NotImplementedError
        
        return is_controllable
    
    def check_is_state_controllable(A, B, x):
        U = CToolbox.get_control_matrix(A, B)
        return np.linalg.matrix_rank(U) == np.linalg.matrix_rank(np.concatenate((U, x.reshape(-1, 1)), axis=1))
    
    def get_controllability_gramian(A, B, t1):
        controllability_gramian = integrate.quad_vec(lambda x: expm(A*x) @ B @ B.T @ expm(A.T*x), 0, t1)[0] 

        return controllability_gramian
    
    def get_observation_matrix(A, C):
        assert len(A.shape) == 2
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] == C.shape[1]

        n = A.shape[0]
        U = np.zeros((0, n))
        for i in range(n):
            new_column = (C @ np.linalg.matrix_power(A, i))
            U = np.concatenate((U, new_column))

        return U
    
    def check_eigenvalues_observable(A, C, method = 'rank_criteria') -> np.array:
        assert len(A.shape) == 2
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] == C.shape[1]
        
        n = A.shape[0]
        is_observable = None

        if method == 'rank_criteria':
            is_observable = []
            eigen_values, _ = np.linalg.eig(A)
            for eigen_value in eigen_values:
                M = np.concatenate((A - eigen_value * np.eye(n), C))
                is_observable.append(np.linalg.matrix_rank(M) == n)
        elif method == 'jordan_form':
            is_observable = []
            P, J = Matrix(A).jordan_form()
            jordan_cells_poses, _ = CToolbox.get_jordan_cells(J)
            C_jordan_form = C @ P
            print("C_jordan_form: ", C_jordan_form)
            for jordan_cells_pose in jordan_cells_poses:
                is_observable.append(True if C_jordan_form[jordan_cells_pose[0]] != 0 else False) 
        else:
            raise NotImplementedError

        return np.array(is_observable)
    
    def check_system_observable(A, C, method = 'rank_criteria'):
        assert len(A.shape) == 2
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] == C.shape[1]
        
        is_observable = None
        n = A.shape[0]
        if method == 'rank_criteria':
            V = CToolbox.get_observation_matrix(A, C)
            is_observable = np.linalg.matrix_rank(V) == n
        elif method == 'eigen_values_criteria':
            is_observable = True
            _, J = Matrix(A).jordan_form()
            _, jordan_cells = CToolbox.get_jordan_cells(J)
        
            for i in range(len(jordan_cells)):
                for j in range(len(jordan_cells)):
                    if i != j and np.array_equal(jordan_cells[i], jordan_cells[j]):
                        is_observable = False
            
            is_observable = is_observable and np.all(CToolbox.check_eigenvalues_observable(A, C))
        else:
            raise NotImplementedError
        
        return is_observable
    
    def get_observability_gramian(A, C, t1):
        observability_gramian = integrate.quad_vec(lambda x: expm(A.T*x) @ C.T @ C @ expm(A*x), 0, t1)[0] 

        return observability_gramian
        
    def get_init_state(A, C, y, t1):
        Q = CToolbox.get_observability_gramian(A, C, t1)
        init_state = np.linalg.pinv(Q) @ integrate.quad_vec(lambda x: expm(A.T*x) @ C.T * y(x), 0, t1)[0]

        return init_state