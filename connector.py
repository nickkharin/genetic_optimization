import zmq


class CoppeliaSimConnector:
    def __init__(self, port=5555):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f'tcp://localhost:{port}')

    def send_request(self, message):
        """Отправляет запрос в CoppeliaSim и возвращает ответ."""
        self.socket.send_string(message)
        return self.socket.recv_string()

    def close(self):
        """Закрывает соединение с CoppeliaSim."""
        self.socket.close()
        self.context.term()

