"""Own one real netplay process and expose public frames and local inputs."""
from collections import OrderedDict
from dataclasses import asdict
import ctypes
from ctypes import wintypes
import ipaddress
import os
import sys
import time

import psutil

import sokurl
from game_runtime.startup import title_configuration
from network_runtime.history import NetworkHistoryClient
from network_runtime.input import NetworkInputClient
from network_runtime.input_events import NetworkInputEventsClient
from soku_rl.play.match import MatchLifecycle
from network_runtime.state import NetworkStateClient
from startup_dialogs import blocking_dialogs
from soku_rl.env.observation.visibility import VisibilityConfig
from soku_rl.env.observation.visible_state import observe_visible_states


class NetworkGame:
    def __init__(self, settings, visibility, timeout, render):
        if set(settings) != {"role", "address", "port", "automate_menu"}:
            raise ValueError("network settings must specify role, address, port and automate_menu")
        if settings["role"] not in ("host", "join") or type(settings["automate_menu"]) is not bool:
            raise ValueError("invalid network role or menu setting")
        ipaddress.IPv4Address(settings["address"])
        if type(settings["port"]) is not int or not 1 <= settings["port"] <= 65535 or timeout <= 0:
            raise ValueError("a valid port and positive launch timeout are required")
        if type(render) is not bool:
            raise ValueError("render must be a boolean")
        self.render = render
        self.settings, self.timeout = settings, timeout
        self.visibility = VisibilityConfig(**visibility)
        self.lifecycle = MatchLifecycle(2)
        self.process = None
        self.peer = None
        self.clients = {}
        self.frames = OrderedDict()
        self.cursor = 0
        self.input_cursor = 0
        self.next_confirm = 0.
        self.result_confirmation = 0
        self.next_dialog_check = 0.
        try:
            self._launch()
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note(f"network cleanup failed: {cleanup_error!r}")
            raise

    def _launch(self):
        with title_configuration(sokurl.SKIPINTRO_INI, self.timeout):
            env = os.environ.copy()
            env.update(SOKURL_VS_BOOTSTRAP="0", SOKURL_UNLIMITED_PACING="0",
                SOKURL_HEADLESS_RENDER="0" if self.render else "1", SOKURL_CAPTURE_IMAGES="0",
                SOKURL_NETWORK_ROLE=self.settings["role"],
                SOKURL_NETWORK_PORT=str(self.settings["port"]),
                SOKURL_NETWORK_ADDRESS=self.settings["address"])
            self.process = psutil.Popen([str(sokurl.GAME_EXE)], cwd=sokurl.GAME_DIR,
                                        env=env, stdout=sys.stderr, stderr=sys.stderr)
            deadline = time.monotonic()+self.timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("network game exited during startup")
                try:
                    scene = sokurl._read_process_values(self.process.pid)[0]
                except OSError:
                    self._check_dialogs()
                    time.sleep(.01)
                    continue
                if scene in (2, 8, 9, 10, 11, 13, 14):
                    self.clients["state"] = NetworkStateClient(self.process.pid)
                    self.clients["history"] = NetworkHistoryClient(self.process.pid)
                    self.clients["input"] = NetworkInputClient(self.process.pid)
                    self.clients["input_events"] = NetworkInputEventsClient(self.process.pid)
                    self.clients["menu"] = sokurl.BridgeClient(self.process.pid)
                    return
                self._check_dialogs()
                time.sleep(.01)
            raise TimeoutError("network game did not reach the title scene")

    def _check_dialogs(self):
        now = time.monotonic()
        if now >= self.next_dialog_check:
            dialogs = blocking_dialogs({self.process.pid})
            if dialogs:
                raise RuntimeError(f"network game blocked by dialogs: {dialogs}")
            self.next_dialog_check = now+2

    def wait_host(self):
        if self.settings["role"] != "host":
            raise ValueError("only a host can wait for its listening socket")
        deadline = time.monotonic()+self.timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise EOFError("network host exited before listening")
            if any(connection.laddr.port == self.settings["port"]
                   for connection in self.process.net_connections(kind="udp")):
                return {"pid": self.process.pid, "port": self.settings["port"]}
            self._check_dialogs()
            time.sleep(.05)
        raise TimeoutError("network host did not open its UDP socket")

    def poll(self):
        exit_code = self.process.poll()
        if exit_code is not None:
            if exit_code != 0:
                raise RuntimeError(f"owned network game failed with exit code {exit_code}")
            return {"closed": True}
        if self.peer is not None and not self.peer.is_running():
            return {"closed": True}
        cursor, snapshots = self.clients["history"].read_after(self.cursor, 2.)
        records = []
        for snapshot in snapshots:
            events = self.lifecycle.update(snapshot.match_state)
            record = {"match": snapshot.match, "round": snapshot.raw.roundId,
                "frame": snapshot.updates, "scene": snapshot.scene, "seat": snapshot.local_seat,
                "scores": snapshot.scores, "phase": self.lifecycle.phase,
                "events": tuple(asdict(event) for event in events)}
            if snapshot.in_battle:
                record["characters"] = (snapshot.raw.p1.characterId, snapshot.raw.p2.characterId)
                if self.settings["automate_menu"]:
                    seat = ("host", "join").index(self.settings["role"])
                    expected = (1, 0)[seat]
                    if record["characters"][seat] != expected:
                        raise RuntimeError(f"AI character differs from training: seat={seat}, "
                                           f"expected={expected}, actual={record['characters'][seat]}")
                record["battle_mode"] = snapshot.raw.battleMode
                record["battle_submode"] = snapshot.raw.battleSubMode
                record["engine_inputs"] = tuple(
                    tuple(getattr(player.input, name) for name, _ in player.input._fields_)
                    for player in (snapshot.raw.p1, snapshot.raw.p2))
                record["render"] = asdict(snapshot.render)
                record["state_hash"] = snapshot.raw.stateHash
            if self.lifecycle.can_act:
                try:
                    record["observations"] = observe_visible_states(snapshot.raw, snapshot.render, self.visibility)
                except Exception as error:
                    raise RuntimeError(f"network visibility failed: metadata={record!r}, "
                                       f"raw_hex={bytes(snapshot.raw).hex()}") from error
                self.frames[snapshot.match, snapshot.updates] = snapshot
                while len(self.frames) > 256:
                    self.frames.popitem(last=False)
            records.append(record)
        self.cursor = cursor
        latest = self.clients["state"].read(2.)
        menu_reply = "not_requested"
        now = time.monotonic()
        if self.result_confirmation:
            reply = self.clients["input"].read_reply(self.result_confirmation)
            if reply != "pending":
                menu_reply = reply
                self.result_confirmation = 0
        if self.settings["automate_menu"] and not self.result_confirmation and now >= self.next_confirm:
            if latest.scene in (8, 9):
                menu = self.clients["menu"]
                if menu.block.commandSeq == menu.block.ackSeq:
                    menu.menu_choose_character(1 if self.settings["role"] == "host" else 0)
                    menu_reply = "selection_requested"
                    # Six render frames between pulses leave a released key and
                    # let the original network menu synchronize each selection.
                    self.next_confirm = now+.1
            elif latest.in_battle and max(latest.scores) >= 2:
                controller = self.clients["input"]
                self.result_confirmation = controller.confirm_result(latest)
                menu_reply = "result_requested"
                self.next_confirm = now+1
        if not latest.in_battle:
            self._check_dialogs()
        self.input_cursor, input_events = self.clients["input_events"].read_after(self.input_cursor, 2.)
        return {"closed": False, "records": records, "cursor": cursor, "input_events": input_events, "menu_reply": menu_reply,
                "pid": self.process.pid}

    def watch_local_peer(self, pid):
        if type(pid) is not int or pid <= 0 or pid == self.process.pid:
            raise ValueError("a distinct local peer process is required")
        self.peer = psutil.Process(pid)
        return {"pid": self.peer.pid}

    def set_caption(self, caption):
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("a nonempty game caption is required")
        user32 = sokurl.user32
        user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.SetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPCWSTR]
        user32.SetWindowTextW.restype = wintypes.BOOL
        changed = []

        @sokurl.WNDENUMPROC
        def window(hwnd, _):
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == self.process.pid:
                kind = ctypes.create_unicode_buffer(128)
                user32.GetClassNameW(hwnd, kind, len(kind))
                if kind.value == "th123_110a":
                    changed.append(bool(user32.SetWindowTextW(hwnd, caption)))
            return True

        if not user32.EnumWindows(window, 0) or not changed or not all(changed):
            raise RuntimeError("could not label the owned game window")
        return {"pid": self.process.pid, "caption": caption}

    def submit(self, match, frame, keys, duration):
        snapshot = self.frames[match, frame]
        controller = self.clients["input"]
        sequence = controller.submit(snapshot, keys, duration)
        return {"request": sequence, "reply": controller.wait_for_reply(sequence, 2.),
                "observed": frame, "target": frame+5,
                "latest_injection": controller.latest_injection(2.)}

    def close(self):
        try:
            # Closing the owned engine releases all held input. A separate input
            # acknowledgment can never be required while it waits for its peer.
            if self.process is not None and self.process.poll() is None:
                sokurl._post_close(self.process.pid)
                try:
                    self.process.wait(timeout=10)
                except psutil.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=10)
        finally:
            for client in self.clients.values():
                client.close()
            self.clients.clear()
