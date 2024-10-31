# 7DOF Manipulator Optimization with Genetic Algorithm and Q-Learning

This project aims to optimize the control and performance of a 7DOF robotic manipulator using a combination of genetic algorithms (GA) and Q-learning. Designed for applications in robotics and automation, this system explores efficient manipulator configurations for reaching target positions, minimizing energy consumption, and improving stability.

## Project Highlights
- **Genetic Algorithm**: The genetic algorithm forms the core of the optimization process, evolving configurations of the manipulator by performing selection, crossover, and mutation. This approach helps find optimal joint configurations to achieve task goals.
- **Q-Learning Integration**: Q-learning is applied to dynamically adjust GA parameters such as mutation rate and population size, enhancing the adaptability and efficiency of the optimization.
- **Forward and Inverse Kinematics**: The project includes functions for both forward and inverse kinematics calculations, enabling precise positioning and manipulation of the 7DOF robotic arm.
- **Energy and Stability Evaluation**: Fitness evaluations incorporate energy efficiency and stability calculations to ensure optimized performance beyond just reaching target positions.

## Architecture
The project is structured to run the optimizations in a simplified Python simulation, with final configurations visualized in CoppeliaSim. Key components include:

- **Manipulator Model** (`manipulator.py`): Defines the robotic arm’s kinematics, dynamics, and stability.
- **Genetic Algorithm Module** (`genetic.py`): Implements core GA operations—selection, crossover, and mutation.
- **Q-Learning Optimizer** (`optimizer.py`): Contains the Q-learning implementation, enabling adaptive control of GA parameters.
- **Control and Simulation** (`main.py`): Integrates GA and Q-learning to drive the optimization process.
- **Utilities** (`utilities.py`): Provides auxiliary functions, including Denavit-Hartenberg transformations for kinematics.

## Usage
This project is ideal for anyone interested in the intersection of genetic algorithms, reinforcement learning, and robotics. It demonstrates a hybrid approach to robotic arm optimization, balancing efficiency, accuracy, and adaptability in complex manipulations.

---

Feel free to contribute by adding new features, improving algorithms, or suggesting enhancements.
