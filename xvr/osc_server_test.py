from pythonosc import dispatcher
from pythonosc import osc_server

if __name__ == "__main__":
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/*", print)

    server = osc_server.ThreadingOSCUDPServer(
        ("localhost", 9001), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()