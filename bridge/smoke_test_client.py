"""Phase 2 smoke test for the FruitFlyBridge mod's socket.

Connects to the mod's local TCP server, pings it, asks for one state
snapshot, and holds "right" for a second so you can watch the character
in-game respond. Run this only after the mod is built and enabled and
you're standing in a world on the Mac's tModLoader client.

Usage: python bridge/smoke_test_client.py
"""
import json
import socket
import time

HOST, PORT = "127.0.0.1", 7778


def request(sock, payload):
    sock.sendall((json.dumps(payload) + "\n").encode())
    line = sock.makefile().readline()
    return json.loads(line)


def main():
    with socket.create_connection((HOST, PORT), timeout=5) as sock:
        print("ping:", request(sock, {"cmd": "ping"}))

        state = request(sock, {"cmd": "get_state"})
        print("get_state:", json.dumps(state, indent=2)[:1000])
        if not state.get("in_world"):
            print("\nNot in a world yet — join FruitFlyWorld, then rerun.")
            return

        print("\nHolding right for 1s, then releasing (watch the character)...")
        print(request(sock, {"cmd": "set_action", "left": False, "right": True, "jump": False, "use": False}))
        time.sleep(1)
        print(request(sock, {"cmd": "set_action", "left": False, "right": False, "jump": False, "use": False}))


if __name__ == "__main__":
    main()
