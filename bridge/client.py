"""Python client for the FruitFlyBridge mod: newline-delimited JSON over a
loopback TCP socket (127.0.0.1:7778), one request and one response per line.
See bridge/README.md for the commands."""
import json
import socket

HOST, PORT = "127.0.0.1", 7778


class Bridge:
    def __init__(self, host=HOST, port=PORT, timeout=5):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.reader = self.sock.makefile()

    def request(self, payload):
        self.sock.sendall((json.dumps(payload) + "\n").encode())
        line = self.reader.readline()
        if not line:
            raise ConnectionError("bridge closed the connection")
        return json.loads(line)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass
