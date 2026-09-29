"""Start a local player and AI in host-first order for either player seat."""
from contextlib import closing


def start_games(stack, config, visibility, directory, worker_factory):
    human_enabled = config["human"]["enabled"]
    if human_enabled:
        player_seat = config["human"]["seat"]
        if type(player_seat) is not int or player_seat not in (1, 2):
            raise ValueError("human.seat must be 1 or 2")
        role = "join" if player_seat == 1 else "host"
    else:
        role = config["network"]["role"]
    if role not in ("host", "join"):
        raise ValueError("network role must be host or join")
    settings = config["network"] | {"role": role}
    runtimes = {"game": config["runtime"]}
    networks = {"game": settings}
    if human_enabled:
        runtimes["human_game"] = config["runtime"] | {"mute_audio": config["human"]["mute_audio"]}
        networks["human_game"] = settings | {"role": "join" if role == "host" else "host",
            "address": "127.0.0.1", "automate_menu": config["human"]["automate_menu"]}
        networks["game"]["address"] = "127.0.0.1"
    order = sorted(networks, key=lambda key: networks[key]["role"] != "host")
    connections, games = {}, {}
    for key in order:
        connection = stack.enter_context(closing(worker_factory(
            log_path=directory / f"{key}-worker.log", **runtimes[key])))
        if connection.identity["kind"] != "network":
            raise ValueError("netplay requires the dedicated network worker")
        connections[key] = connection
        games[key] = connection.request("start", {"network": networks[key], "visibility": visibility})
        if human_enabled:
            caption = "SokuRL - AI" if key == "game" else "SokuRL - Player"
            connection.request("set_caption", {"caption": caption})
            if networks[key]["role"] == "host":
                connection.request("wait_host", {})
    if human_enabled:
        connections["game"].request("watch_local_peer", {"pid": games["human_game"]["pid"]})
    return connections["game"], games
