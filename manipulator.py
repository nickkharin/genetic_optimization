import numpy as np
import random
from scipy.optimize import minimize
from utilities import *


class Manipulator7DOF:
    def __init__(self, joint_angles=None, lengths=None):
        self.joint_angles = joint_angles if joint_angles is not None else [random.uniform(-np.pi, np.pi) for _ in range(7)]
        self.lengths = lengths if lengths is not None else [random.uniform(0.5, 2.0) for _ in range(7)]

    def forward_kinematics(self):
        T = np.eye(4)  # Инициализация единичной матрицы
        positions = []  # Список для хранения положений звеньев
        for i in range(7):
            # Применение преобразования Денавита-Хартенберга для каждого звена
            theta = self.joint_angles[i]
            a, alpha, d = self.lengths[i], 0, 0
            T = np.dot(T, dh_transform(a, alpha, d, theta))
            positions.append(T[:3, 3])  # Добавление положения звена в список
        return positions

    def inverse_kinematics(self, target_position):
        initial_joint_angles = np.array(self.joint_angles)
        result = minimize(self.objective_function_for_ik, initial_joint_angles, args=(target_position,), method='Nelder-Mead')
        if result.success:
            return result.x
        else:
            raise ValueError("Inverse kinematics solution not found.")

    def objective_function_for_ik(self, joint_angles, target_position):
        self.joint_angles = joint_angles
        current_position = self.forward_kinematics()[-1]
        return np.linalg.norm(current_position - target_position)

    def calculate_dynamics(self, joint_accelerations):
        g = 9.81  # Ускорение свободного падения
        forces = []  # Список для хранения сил, действующих на звенья
        for i, acceleration in enumerate(joint_accelerations):
            # Вычисление силы с учетом гравитации
            force = self.link_masses[i] * (acceleration + g)
            forces.append(force)
        return forces

    def calculate_center_of_mass(self):
        positions = self.forward_kinematics()
        total_mass = len(self.lengths)  # Простое предположение: масса каждого звена = 1
        center_of_mass = sum(positions) / total_mass  # Среднее положение всех звеньев
        return center_of_mass

    def evaluate_load_distribution(self):
        load_variability = np.std(self.joint_angles)  # Стандартное отклонение углов суставов как мера неравномерности
        return 1 / (1 + load_variability)

    def evaluate_dynamic_stability(self, joint_velocities=None, joint_accelerations=None):
        if joint_velocities is None:
            joint_velocities = np.zeros(len(self.joint_angles))
        if joint_accelerations is None:
            joint_accelerations = np.zeros(len(self.joint_angles))

        velocity_variability = np.std(joint_velocities)
        acceleration_variability = np.std(joint_accelerations)

        stability_score = 1 / (1 + velocity_variability + acceleration_variability)
        return stability_score

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
