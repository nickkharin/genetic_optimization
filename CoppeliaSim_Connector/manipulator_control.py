import time
from CoppeliaSim_Connector.connector import CoppeliaSimConnector


class ManipulatorControl:
    def __init__(self, connector):
        self.connector = connector
        self.joint_handles = self._get_joint_handles()


    def _get_joint_handles(self):
        handles = []
        for i in range(7):
            joint_name = f"/redundantRobot/joint{i}"
            handle = self.connector.send_request(f'getObjectHandle:{joint_name}')
            if handle.isdigit():
                handles.append(int(handle))
            else:
                print(f"Failed to get handle for {joint_name}")
        return handles

    def set_joint_angles(self, angles):
        for handle, angle in zip(self.joint_handles, angles):
            self.connector.send_request(f'setJointTargetPosition:{handle},{angle}')

    def set_sphere_position(self, x, y, z):
        handle = self.connector.send_request(f'getObjectHandle:/Sphere')
        position_command = f'setObjectPosition:{handle},{x},{y},{z}'
        response = self.connector.send_request(position_command)

    def start_simulation(self):
        self.connector.send_request('startSimulation')

    def stop_simulation(self):
        self.connector.send_request('stopSimulation')

    def step_simulation(self):
        """Выполняет один шаг симуляции."""
        self.connector.send_request('stepSimulation')

if __name__ == "__main__":
    connector = CoppeliaSimConnector()
    manipulator = ManipulatorControl(connector)
    try:
        manipulator.set_sphere_position(2, 2, 0.2)
        manipulator.set_joint_angles([10, -1.3915318459609356, -0.36028889825648425, 2.9975299740715453, 1.8651015800844783, -1.380450284243989, 142.7434757298145])
        print("Углы соединений установлены.")
        manipulator.start_simulation()
        print("Симуляция запущена.")
        time.sleep(5)
        print("Симуляция выполняется...")
        manipulator.stop_simulation()
        print("Симуляция остановлена.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    finally:
        connector.close()
        print("Соединение закрыто.")

