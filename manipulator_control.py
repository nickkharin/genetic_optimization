# manipulator_control.py

from connector import CoppeliaSimConnector


class ManipulatorControl:
    def __init__(self, connector):
        self.connector = connector
        self.joint_handles = self._get_joint_handles()

    def _get_joint_handles(self):
        """Получает и сохраняет дескрипторы всех соединений манипулятора."""
        handles = []
        for i in range(1, 8):  # Предполагается, что у манипулятора 7 соединений
            handle = self.connector.send_request(f'getObjectHandle:joint{i}')
            handles.append(int(handle))
        return handles

    def set_joint_angles(self, angles):
        """Устанавливает углы для соединений манипулятора."""
        for handle, angle in zip(self.joint_handles, angles):
            self.connector.send_request(f'setJointTargetPosition:{handle},{angle}')

    def start_simulation(self):
        """Запускает симуляцию."""
        self.connector.send_request('startSimulation')

    def stop_simulation(self):
        """Останавливает симуляцию."""
        self.connector.send_request('stopSimulation')

    def step_simulation(self):
        """Выполняет один шаг симуляции."""
        self.connector.send_request('stepSimulation')


# Пример использования
if __name__ == "__main__":
    connector = CoppeliaSimConnector()
    manipulator = ManipulatorControl(connector)
    manipulator.set_joint_angles([0, 0.5, 0, -0.5, 0, 0.5, 0])
    manipulator.start_simulation()
    import time

    time.sleep(5)
    manipulator.stop_simulation()
    connector.close()
