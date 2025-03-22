import numpy as np
import random
from scipy.optimize import minimize, Bounds
from utilities import dh_transform
from functools import lru_cache
import logging

class Manipulator7DOF:
    """
    Класс, представляющий 7-степенный манипулятор.

    Учтены более реалистичные углы суставов (joint limits) и
    расстановка DH-параметров, ориентировочно соответствующая
    роботу на скриншоте из CoppeliaSim.
    """

    def __init__(
        self,
        joint_angles=None,
        lengths=None,
        link_masses=None,
        inertia_tensors=None,
        joint_frictions=None,
        alphas=None,
        ds=None,
        environment_temp=20
    ):
        """
        Инициализация манипулятора с возможностью задания параметров DH.
        Параметры:
          - joint_angles: начальные углы каждого из 7 суставов (рад).
          - lengths: длины звеньев (a_i) для каждого из 7 звеньев.
          - alphas: твисты звеньев (alpha_i).
          - ds: смещения (d_i).
          - environment_temp: температура окружения (непосредственно не используется).
        """

        # Если длины не переданы, даём некие усреднённые длины звеньев
        if lengths is None:
            lengths = [0.2, 0.4, 0.3, 0.35, 0.3, 0.2, 0.2]

        # Начальные углы: либо переданные, либо случайно от -π..+π
        if joint_angles is None:
            joint_angles = [random.uniform(-np.pi, np.pi) for _ in range(7)]

        # Примерный твист (alpha)
        if alphas is None:
            alphas = [
                0.0,         # joint0
                np.pi/2,     # joint1
                0.0,         # joint2
                -np.pi/2,    # joint3
                np.pi/2,     # joint4
                -np.pi/2,    # joint5
                0.0          # joint6
            ]

        # Примерные смещения (d)
        if ds is None:
            ds = [
                0.3,  # небольшое смещение по оси первого звена
                0.0,
                0.4,
                0.0,
                0.3,
                0.0,
                0.2
            ]

        self.lengths = lengths
        self.joint_angles = joint_angles
        self.alphas = alphas
        self.ds = ds
        self.link_masses = link_masses if link_masses is not None else [random.uniform(1.0, 5.0) for _ in range(7)]
        self.inertia_tensors = inertia_tensors if inertia_tensors is not None else [
            np.eye(3) * random.uniform(0.1, 1.0) for _ in range(7)
        ]
        self.joint_frictions = joint_frictions if joint_frictions is not None else [
            random.uniform(0.01, 0.05) for _ in range(7)
        ]
        self.environment_temp = environment_temp

        # Примерные ограничения углов (рад):
        self.joint_limits = [
            (-np.pi, np.pi),        # joint0
            (-np.pi/2, np.pi/2),    # joint1
            (-np.pi, np.pi),        # joint2
            (-np.pi, np.pi),        # joint3
            (-np.pi, np.pi),        # joint4
            (-np.pi, np.pi),        # joint5
            (-np.pi, np.pi)         # joint6
        ]

        # Хранение траектории (последовательность позиций звеньев)
        self.trajectory = []

        # Применяем клип к начальному состоянию
        self._clip_joint_angles()
        self.log_joint_angles()

    def _clip_joint_angles(self):
        """
        Вспомогательный метод: обрезает каждое значение угла
        по соответствующим границам self.joint_limits.
        """
        clipped = []
        for i, angle in enumerate(self.joint_angles):
            lo, hi = self.joint_limits[i]
            clipped.append(np.clip(angle, lo, hi))
        self.joint_angles = clipped

    def get_joint_angles(self):
        """
        Возвращает текущие углы суставов манипулятора.
        """
        return self.joint_angles

    def set_joint_angles(self, angles):
        """
        Устанавливает углы суставов манипулятора (с учётом ограничений).
        """
        if len(angles) != 7:
            raise ValueError("Количество углов должно быть 7 для 7DoF манипулятора.")

        self.joint_angles = angles
        self._clip_joint_angles()

        self.log_joint_angles()
        self.log_trajectory()

    def log_trajectory(self):
        """
        Логирует текущую траекторию манипулятора:
        сохраняет полный список координат от базы до конца эффектора.
        """
        positions = self.forward_kinematics()
        self.trajectory.append(positions)

    def get_trajectory(self):
        """
        Возвращает сохранённую траекторию.
        """
        return self.trajectory

    def reset_trajectory(self):
        """
        Сбрасывает сохранённую траекторию.
        """
        self.trajectory = []

    def max_reach(self):
        """
        Рассчитывает максимальную досягаемость (сумма длины звеньев).
        """
        return sum(self.lengths)

    def forward_kinematics(self):
        """
        Вычисляет позицию каждого сустава (и конец эффектора) по DH-параметрам.
        Возвращает список из 8 точек (x, y, z):
          - (0,0,0) для базы
          - 7 промежуточных точек после каждого звена.
        """
        T = np.eye(4)
        positions = [(0.0, 0.0, 0.0)]  # Начальная точка - база

        for i in range(7):
            theta = self.joint_angles[i]
            a = self.lengths[i]
            alpha = self.alphas[i]
            d = self.ds[i]

            # Матрица Денавита–Хартенберга
            T = T @ dh_transform(a, alpha, d, theta)

            x, y, z = T[0, 3], T[1, 3], T[2, 3]
            positions.append((x, y, z))

        return positions

    @lru_cache(maxsize=32)
    def inverse_kinematics_multiple_solutions(self, target_position, num_trials=10):
        """
        Генерирует несколько решений обратной кинематики и выбирает наилучшее.
        """
        solutions = []
        lower_bounds = [lim[0] for lim in self.joint_limits]
        upper_bounds = [lim[1] for lim in self.joint_limits]
        bounds = Bounds(lower_bounds, upper_bounds)

        for _ in range(num_trials):
            initial_joint_angles = np.array([
                random.uniform(self.joint_limits[i][0], self.joint_limits[i][1])
                for i in range(7)
            ])
            result = minimize(
                self.objective_function_for_ik,
                initial_joint_angles,
                args=(target_position,),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 150, 'disp': False}
            )
            if result.success:
                solutions.append(result.x)
            else:
                logging.warning("Оптимизация (IK) не нашла решения.")

        if solutions:
            return self.evaluate_solutions(solutions, target_position)
        else:
            raise ValueError("Не найдено ни одного решения обратной кинематики.")

    def objective_function_for_ik(self, proposed_angles, target_position):
        """
        Целевая функция для обратной кинематики.
        Важно: не изменяем self.joint_angles глобально,
        а лишь временно подставляем, чтобы вычислить расстояние.
        """
        # Сохраняем текущее состояние
        original_angles = self.joint_angles[:]

        # Клиппируем входные углы
        clipped_angles = []
        for i, angle in enumerate(proposed_angles):
            lo, hi = self.joint_limits[i]
            clipped_angles.append(np.clip(angle, lo, hi))

        # Применяем локально
        self.joint_angles = clipped_angles

        # Считаем дистанцию
        current_position = self.forward_kinematics()[-1]
        distance = np.linalg.norm(np.array(current_position) - np.array(target_position))

        # Восстанавливаем прежние углы
        self.joint_angles = original_angles

        return distance

    def calculate_center_of_mass(self):
        """
        Вычисляет центр масс манипулятора как средневзвешенное
        положение всех звеньев (кроме базы).
        """
        positions = self.forward_kinematics()[1:]
        total_mass = sum(self.link_masses)
        center_of_mass = np.sum(
            np.array(positions) * np.array(self.link_masses)[:, None],
            axis=0
        ) / total_mass
        return center_of_mass

    def energy_consumption(self, joint_angles=None):
        """
        Простая модель энергопотребления: пропорциональна сумме |углов|.
        """
        if joint_angles is None:
            joint_angles = self.joint_angles
        return sum(abs(a) for a in joint_angles) * 0.1

    def evaluate_stability(self):
        """
        Комплексная оценка стабильности:
        1 / (1 + расстояние COM от (0,0,0) + разброс углов).
        """
        com_score = np.linalg.norm(self.calculate_center_of_mass())
        joint_variability = np.std(self.joint_angles)
        return 1.0 / (1.0 + com_score + joint_variability)

    def evaluate_solutions(self, solutions, target_position):
        """
        Выбирает минимальную objective_function_for_ik.
        """
        best_solution = min(
            solutions,
            key=lambda angles: self.objective_function_for_ik(angles, target_position)
        )
        return best_solution

    def reset(self):
        """
        Сбрасывает углы суставов к 0 и очищает траекторию.
        """
        self.joint_angles = [0.0] * 7
        self._clip_joint_angles()
        self.log_joint_angles()
        self.reset_trajectory()

    def log_joint_angles(self):
        """
        Логирует текущие углы суставов (радианы) на уровне debug,
        чтобы не засорять основной лог.
        """
        logging.debug(f"[Manipulator7DOF] Current Joint Angles: {self.joint_angles}")

    def __str__(self):
        """
        Строковое представление.
        """
        return f"Manipulator Angles: {self.joint_angles}, Lengths: {self.lengths}"