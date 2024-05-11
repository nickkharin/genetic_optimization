import numpy as np
import random
from manipulator import *


def crossover(parent1, parent2):
    child = Manipulator7DOF()
    for i in range(len(child.joint_angles)):
        child.joint_angles[i] = random.choice([parent1.joint_angles[i], parent2.joint_angles[i]])
        child.lengths[i] = random.choice([parent1.lengths[i], parent2.lengths[i]])
    return child

def mutate(robot, mutation_rate=0.1):
    for i in range(len(robot.joint_angles)):
        if random.random() < mutation_rate:
            robot.joint_angles[i] += random.uniform(-0.1, 0.1)
            robot.lengths[i] += random.uniform(-0.1, 0.1)
            # Обеспечиваем, что длина звена не станет отрицательной или слишком маленькой
            robot.lengths[i] = max(0.1, robot.lengths[i])
    return robot

def select_parents(population, fitness_scores, num_parents):
    # Сортировка популяции по оценкам приспособленности (от лучших к худшим)
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0])]
    selected_parents = sorted_population[:num_parents]
    return selected_parents

def generate_initial_population(pop_size):
    population = [Manipulator7DOF() for _ in range(pop_size)]
    return population

def multi_criteria_fitness(robot, target, joint_velocities=np.zeros(7), joint_accelerations=np.zeros(7)):
    distance_score = robot.evaluate_distance(target)  # Наказание за расстояние до цели
    stability_score = robot.evaluate_stability(joint_velocities, joint_accelerations)  # Награда за стабильность
    energy_score = robot.energy_consumption()  # Наказание за энергопотребление
    # Общая оценка: меньшее значение означает лучшую приспособленность
    return distance_score + stability_score + energy_score