"""Check error-dialog text and process ownership against the Windows API."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest


@unittest.skipUnless(os.name == "nt", "requires Windows dialog APIs")
class StartupDialogTests(unittest.TestCase):
    def test_reads_only_the_owned_process_dialog(self):
        path = Path(__file__).parents[1] / "tools/startup_dialogs.py"
        spec = importlib.util.spec_from_file_location("startup_dialogs", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        script = (
            "import ctypes; from ctypes import wintypes; "
            "show=ctypes.WinDLL('user32').MessageBoxW; "
            "show.argtypes=[wintypes.HWND,wintypes.LPCWSTR,wintypes.LPCWSTR,wintypes.UINT]; "
            "show(None,'Owned startup diagnostic','DSound-Error',0)"
        )
        # Bypass a venv launcher so the returned PID owns the dialog.
        child = subprocess.Popen([sys._base_executable, "-c", script])
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                self.assertIsNone(child.poll())
                dialogs = module.blocking_dialogs({child.pid})
                if dialogs:
                    break
                time.sleep(.05)
            self.assertEqual(len(dialogs), 1)
            self.assertEqual(dialogs[0]["pid"], child.pid)
            self.assertEqual(dialogs[0]["title"], "DSound-Error")
            self.assertIn("Owned startup diagnostic", dialogs[0]["messages"])
            self.assertEqual(module.blocking_dialogs({os.getpid()}), [])
        finally:
            child.terminate()
            child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
