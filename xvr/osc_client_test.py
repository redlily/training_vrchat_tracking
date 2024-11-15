import random
import time

from pythonosc import udp_client

def main():
    client = udp_client.SimpleUDPClient("127.0.0.1", 9000)

    for x in range(1000):
        client.send_message(f"/tracking/trackers/1/position", [-0.125, 0.0, 0.0]) # right feet
        client.send_message(f"/tracking/trackers/1/rotation", [0.0, 0.0, 0.0])
        client.send_message(f"/tracking/trackers/2/position", [0.125, 0.0, 0.0]) # left feet
        client.send_message(f"/tracking/trackers/2/rotation", [0.0, 0.0, 0.0])

        client.send_message(f"/tracking/trackers/3/position", [-0.125, 0.55, 0.0]) # right knees
        client.send_message(f"/tracking/trackers/3/rotation", [0.0, 0.0, 0.0])
        client.send_message(f"/tracking/trackers/4/position", [0.125, 0.55, 0.0]) # left knees
        client.send_message(f"/tracking/trackers/4/rotation", [0.0, 0.0, 0.0])

        client.send_message(f"/tracking/trackers/5/position", [0.0, 0.85, 0.0]) # hip
        client.send_message(f"/tracking/trackers/5/rotation", [0.0, 0.0, 0.0])
        client.send_message(f"/tracking/trackers/6/position", [0.0, 1.35, 0.0]) # chest
        client.send_message(f"/tracking/trackers/6/rotation", [0.0, 0.0, 0.0])

        client.send_message(f"/tracking/trackers/7/position", [-0.4, 1.5, 0.0]) # right elbows
        client.send_message(f"/tracking/trackers/7/rotation", [0.0, 0.0, 0.0])
        client.send_message(f"/tracking/trackers/8/position", [0.4, 1.5, 0.0]) # left elbows
        client.send_message(f"/tracking/trackers/8/rotation", [0.0, 0.0, 0.0])

        time.sleep(1)
        print("hoge")

if __name__ == "__main__":
    print(f"10 + 1 = {10 + 1}")
    main()