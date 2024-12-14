import numpy as np
import random
from scipy.optimize import minimize, Bounds
from utilities import dh_transform
from functools import lru_cache
import logging

class Manipulator7DOF:
    """
    Класс, представляющий 7-степенный манипулятор.
    """

    def __init__(self, joint_angles=None, lengths=None, link_masses=None, inertia_tensors=None, joint_frictions=None, alphas=None, ds=None, environment_temp=20):
        """
        Инициализация манипулятора с возможностью задания параметров DH.
        """
        self.lengths = lengths if lengths is not None else [random.uniform(0.5, 2.0) for _ in range(7)]
        self.joint_angles = joint_angles if joint_angles is not None else [random.uniform(-np.pi, np.pi) for _ in range(len(self.lengths))]
        self.alphas = alphas if alphas is not None else [random.uniform(-np.pi / 2, np.pi / 2) for _ in range(7)]  # Увеличен диапазон
        self.ds = ds if ds is not None else [random.uniform(0.1, 1.0) for _ in range(7)]  # Увеличен диапазон
        self.link_masses = link_masses if link_masses is not None else [random.uniform(1.0, 5.0) for _ in range(7)]
        self.inertia_tensors = inertia_tensors if inertia_tensors is not None else [np.eye(3) * random.uniform(0.1, 1.0) for _ in range(7)]
        self.joint_frictions = joint_frictions if joint_frictions is not None else [random.uniform(0.01, 0.05) for _ in range(7)]
        self.environment_temp = environment_temp  # Температура окружающей среды

    def max_reach(self):
        """
        Рассчитывает максимальную досягаемость манипулятора.
        """
        return sum(self.lengths)

    def forward_kinematics(self):
        """
        Вычисляет положение каждого сустава с помощью прямой кинематики.
        """
        T = np.eye(4)
        positions = []
        for i in range(7):
            theta = self.joint_angles[i]
            a = self.lengths[i]
            alpha = self.alphas[i]
            d = self.ds[i]
            T = np.dot(T, dh_transform(a, alpha, d, theta))
            # Берём XYZ-координаты
            positions.append((T[0, 3], T[1, 3], T[2, 3]))
        return positions

    @lru_cache(maxsize=32)
    def inverse_kinematics_multiple_solutions(self, target_position, num_trials=10):
        """
        Генерирует несколько решений обратной кинематики и выбирает наилучшее.
        """
        solutions = []
        bounds = Bounds([-np.pi] * 7, [np.pi] * 7)
        for _ in range(num_trials):
            initial_joint_angles = np.array([random.uniform(-np.pi, np.pi) for _ in range(7)])
            result = minimize(
                self.objective_function_for_ik,
                initial_joint_angles,
                args=(target_position,),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 100, 'disp': False}
            )
            if result.success:
                solutions.append(result.x)
            else:
                logging.warning("Optimization failed during IK computation.")

        if solutions:
            return self.evaluate_solutions(solutions, target_position)
        else:
            raise ValueError("No valid inverse kinematics solution found.")

    def objective_function_for_ik(self, joint_angles, target_position):
        """
        Целевая функция для обратной кинематики.
        """
        self.joint_angles = joint_angles
        current_position = self.forward_kinematics()[-1]  # Положение конца манипулятора
        return np.linalg.norm(np.array(current_position) - np.array(target_position))  # Евклидово расстояние до цели

    def calculate_center_of_mass(self):
        """
        Вычисляет центр масс манипулятора.
        """
        positions = self.forward_kinematics()
        total_mass = sum(self.link_masses)
        center_of_mass = np.sum(np.array(positions) * np.array(self.link_masses)[:, None], axis=0) / total_mass
        return center_of_mass

    def energy_consumption(self, joint_angles=None):
        """
        Вычисляет энергопотребление манипулятора.
        """
        if joint_angles is None:
            joint_angles = self.joint_angles
        return sum(abs(angle) for angle in joint_angles) * 0.1  # Простая модель энергопотребления

    def evaluate_stability(self):
        """
        Комплексная оценка стабильности манипулятора.
        """
        com_score = np.linalg.norm(self.calculate_center_of_mass())  # Положение центра масс
        joint_variability = np.std(self.joint_angles)  # Вариативность углов суставов
        return 1 / (1 + com_score + joint_variability)  # Итоговая стабильность

    def evaluate_solutions(self, solutions, target_position):
        """
        Выбирает наилучшее решение из предложенных.
        """
        best_solution = min(solutions, key=lambda x: self.objective_function_for_ik(x, target_position))
        return best_solution

    def reset(self):
        """
        Сбрасывает углы суставов манипулятора к начальному состоянию.
        """
        self.joint_angles = np.zeros(7)

    def __str__(self):
        """
        Вывод информации о манипуляторе.
        """
        return f"Manipulator Angles: {self.joint_angles}, Lengths: {self.lengths}"