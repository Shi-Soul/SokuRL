"""Resolve original script bytes and fingerprint the complete external package."""
import hashlib
from pathlib import Path


class ScriptPackage:
    def __init__(self, directory, api_source):
        self.root = Path(directory).resolve(strict=True)
        self.api = Path(api_source).read_bytes()
        self.files, self.aliases = {}, {}
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.root).as_posix()
            self.files[relative] = path.read_bytes()
            for encoding in ("utf-8", "cp932", "gbk"):
                try:
                    alias = relative.encode(encoding)
                except UnicodeEncodeError:
                    continue
                if alias in self.aliases and self.aliases[alias] != relative:
                    raise ValueError("ambiguous original script filename encoding")
                self.aliases[alias] = relative
        if not self.files or b"function api_main" not in self.api:
            raise ValueError("the complete script directory and original api.ai are required")
        digest = hashlib.sha256(self.api)
        for name, data in self.files.items():
            digest.update(name.encode("utf-8") + b"\0" + data)
        self.fingerprint = digest.hexdigest()

    def source(self, name):
        if isinstance(name, str):
            name = name.encode("utf-8")
        name = name.replace(b"\\", b"/")
        if name not in self.aliases:
            raise FileNotFoundError(f"script is absent from the source package: {name!r}")
        return self.files[self.aliases[name]]

    def character_script(self, character):
        names = [name for name in self.files if name.startswith(f"{character:02}_")
                 and name.endswith("_main.ai")]
        if len(names) != 1:
            raise ValueError("each character must have exactly one canonical main script")
        return names[0]
