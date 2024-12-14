import zmq


class CoppeliaSimConnector:
    def __init__(self, port=5555):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f'tcp://localhost:{port}')

    def send_request(self, message):
        self.socket.send_string(message)
        return self.socket.recv_string()

    def test_connection(self):
        try:
            response = self.send_request('getVersion')
            print(f"Connected to CoppeliaSim, version {response}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def close(self):
        self.socket.close()
        self.context.term()




