import numpy as np
import random
from scipy.optimize import minimize, Bounds
from utilities import *
from functools import lru_cache

class Manipulator7DOF:
    def __init__(self, joint_angles=None, lengths=None, link_masses=None, inertia_tensors=None):
        self.joint_angles = joint_angles if joint_angles is not None else [random.uniform(-np.pi, np.pi) for _ in range(7)]
        self.lengths = lengths if lengths is not None else [random.uniform(0.5, 2.0) for _ in range(7)]
        self.link_masses = link_masses if link_masses is not None else [random.uniform(1.0, 5.0) for _ in range(7)]
        self.inertia_tensors = inertia_tensors if inertia_tensors is not None else [np.eye(3) * random.uniform(0.1, 1.0) for _ in range(7)]

    def forward_kinematics(self):
        """ Calculate the position of each joint using forward kinematics. """
        T = np.eye(4)  # Identity matrix initialization
        positions = []
        for i in range(7):
            theta = self.joint_angles[i]
            a, alpha, d = self.lengths[i], 0, 0
            T = np.dot(T, dh_transform(a, alpha, d, theta))
            positions.append(T[:3, 3])
        return positions

    @lru_cache(maxsize=32)
    def inverse_kinematics_multiple_solutions(self, target_position, num_trials=10):
        """ Generate multiple solutions for inverse kinematics and select the best one. """
        solutions = []
        for _ in range(num_trials):
            initial_joint_angles = np.array([random.uniform(-np.pi, np.pi) for _ in range(7)])
            bounds = Bounds([-np.pi] * 7, [np.pi] * 7)
            result = minimize(self.objective_function_for_ik, initial_joint_angles, args=(target_position,),
                              method='SLSQP', bounds=bounds, options={'maxiter': 100, 'disp': False})
            if result.success:
                solutions.append(result.x)

        # Evaluate all successful solutions and select the best one
        if solutions:
            best_solution = self.evaluate_solutions(solutions)
            return best_solution
        else:
            raise ValueError("No valid inverse kinematics solution found.")

    def objective_function_for_ik(self, joint_angles, target_position):
        """ Objective function for the minimization in inverse kinematics. """
        self.joint_angles = joint_angles
        current_position = self.forward_kinematics()[-1]
        return np.linalg.norm(current_position - target_position)

    def calculate_dynamics(self, joint_velocities, joint_accelerations):
        """ Calculate forces and torques on each joint including inertial, Coriolis, and centrifugal effects. """
        # Calculate gravitational forces
        g = 9.81  # Acceleration due to gravity
        gravitational_forces = [self.link_masses[i] * g for i in range(7)]

        # Calculate inertial forces
        inertial_forces = [np.dot(self.inertia_tensors[i], joint_accelerations[i]) for i in range(7)]

        # Calculate Coriolis and centrifugal forces
        coriolis_and_centrifugal_forces = self.calculate_coriolis_and_centrifugal_forces(joint_velocities,
                                                                                         joint_accelerations)

        # Total forces and torques
        total_forces = np.add(np.add(gravitational_forces, inertial_forces), coriolis_and_centrifugal_forces)
        return total_forces

    def calculate_coriolis_and_centrifugal_forces(self, joint_velocities, joint_accelerations):
        """ Calculate Coriolis and centrifugal forces based on the robot's velocity and acceleration dynamics. """
        # Placeholder for the Coriolis and centrifugal force calculation (requires Jacobian matrices)
        # This is a simplified placeholder, and you would need to calculate this properly using the robot's dynamics
        J = self.calculate_jacobian()
        # For simplicity, assuming J_dot * q_dot is zero
        coriolis_and_centrifugal = np.dot(J.T, joint_velocities) * joint_accelerations
        return coriolis_and_centrifugal

    def calculate_jacobian(self):
        """ Calculate the Jacobian matrix for the manipulator. """
        # This is a simplified placeholder calculation for the Jacobian
        # A proper implementation should calculate the partial derivatives of the end-effector's position with respect to each joint variable
        J = np.zeros((6, 7))  # Assuming a 6x7 Jacobian for a 7 DOF manipulator
        return J

    def calculate_center_of_mass(self):
        """ Calculate the center of mass of the manipulator. """
        positions = self.forward_kinematics()
        total_mass = sum(self.link_masses)
        center_of_mass = np.dot(self.link_masses, positions) / total_mass
        return center_of_mass

    def evaluate_solutions(self, solutions):
        """ Evaluate multiple IK solutions and select the best based on a criterion. """
        min_energy = float('inf')
        best_solution = None
        for solution in solutions:
            energy = self.energy_consumption(solution)
            if energy < min_energy:
                min_energy = energy
                best_solution = solution
        return best_solution

    def evaluate_load_distribution(self):
        """ Evaluate the distribution of load across the joints. """
        load_variability = np.std(self.joint_angles)
        return 1 / (1 + load_variability)

    def evaluate_dynamic_stability(self, joint_velocities=None, joint_accelerations=None):
        if joint_velocities is None:
            joint_velocities = np.zeros(len(self.joint_angles))
        if joint_accelerations is None:
            joint_accelerations = np.zeros(len(self.joint_angles))

        velocity_variability = np.std(joint_velocities)
        acceleration_variability = np.std(joint_accelerations)
        return 1 / (1 + velocity_variability + acceleration_variability)

    def evaluate_stability(self, joint_velocities=np.zeros(7), joint_accelerations=np.zeros(7)):
        com_score = np.linalg.norm(self.calculate_center_of_mass())
        load_distribution_score = self.evaluate_load_distribution()
        dynamic_stability_score = self.evaluate_dynamic_stability(joint_velocities, joint_accelerations)
        return com_score + load_distribution_score + dynamic_stability_score

    def evaluate_distance(self, target):
        positions = self.forward_kinematics()
        end_effector_pos = positions[-1]
        return np.linalg.norm(end_effector_pos - np.array(target))

    def energy_consumption(self):
        return sum(abs(angle) for angle in self.joint_angles) * 0.1

    def __str__(self):
        return f'Manipulator Angles: {self.joint_angles}, \n Lengths: {self.lengths}'
