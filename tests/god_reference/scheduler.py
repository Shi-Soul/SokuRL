"""Run the unchanged api.ai scheduler on a thread with a blocking frame boundary."""
import queue
import threading

from lupa.lua51 import LuaRuntime
from soku_rl.policy.god.api import ScriptAPI


class OriginalScheduler:
    def __init__(self, package, script, seed, first):
        self.requests, self.results = queue.Queue(), queue.Queue()
        self.package, self.script, self.seed, self.first = package, script, seed, first
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        lua = LuaRuntime(encoding=None, unpack_returned_tuples=True)
        api = ScriptAPI(lua, self.seed)

        def boundary():
            try:
                self.results.put(("inputs", api.inputs()))
                message = self.requests.get(timeout=30)
                if message == "stop":
                    lua.globals()[b"thread_num"] = 0
                else:
                    api.observe(message)
            except BaseException as error:
                self.results.put(("error", repr(error)))
                lua.globals()[b"thread_num"] = 0

        def require(name):
            lua.execute(self.package.source(name), name=b"@original/" + name)

        lua.globals()[b"_yield"] = boundary
        lua.globals()[b"require"] = require
        lua.globals()[b"print"] = lambda *parts: None
        try:
            lua.execute(self.package.api)
            api.observe(self.first)
            require(self.script.encode("utf-8"))
            lua.globals()[b"api_main"]()
            self.results.put(("stopped", None))
        except BaseException as error:
            self.results.put(("error", repr(error)))

    def inputs(self):
        kind, value = self.results.get(timeout=30)
        if kind != "inputs":
            raise AssertionError((kind, value))
        return value

    def advance(self, observation):
        self.requests.put(observation)

    def close(self):
        self.requests.put("stop")
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("original scheduler failed to stop")
