"""Read each request result and its terminal scheduling event in order."""
import struct

from network_history import NetworkHistoryReader
from network_input import RESULTS


EVENT = struct.Struct("<6I3Q")
KINDS = RESULTS | {8: "injected", 9: "expired", 10: "cancelled", 11: "superseded"}


def decode_input_event(data):
    if len(data) != EVENT.size:
        raise ValueError("invalid network input event length")
    request, kind, command, match, round_id, duration, observed, target, at = EVENT.unpack(data)
    if not request or kind not in KINDS:
        raise ValueError("invalid network input event identity")
    return {"request": request, "result": KINDS[kind], "command_type": command,
            "match": match, "round": round_id, "duration": duration,
            "observed": observed, "target": target, "at": at}


class NetworkInputEventsClient(NetworkHistoryReader):
    mapping = "NetworkInputEvents"
    magic = 0x454E4B53
    entry_size = EVENT.size
    decode = staticmethod(decode_input_event)
