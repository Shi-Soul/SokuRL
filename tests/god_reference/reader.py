"""Call independently compiled upstream functions without injecting a game DLL."""
import ctypes as C


MEMORY_READ = C.CFUNCTYPE(C.c_int, C.c_uint32, C.c_void_p, C.c_uint32)


class Reference:
    def __init__(self, library):
        self.dll = C.CDLL(str(library))
        signatures = {
            "init": ([MEMORY_READ], None),
            "reload": ([C.c_uint32, C.c_int, C.c_int], None),
            "global": ([C.c_char_p], C.c_double),
            "entity": ([C.c_int, C.c_int, C.POINTER(C.c_double)], None),
            "box": ([C.c_int] * 4 + [C.POINTER(C.c_int)], None),
            "field": ([C.c_int] * 3, C.c_int),
            "keys": ([C.c_int] * 2, None),
            "key": ([C.c_int] * 2, C.c_int),
            "delayed_action": ([C.c_int] * 4, None),
            "projectiles": ([C.c_float, C.c_int, C.POINTER(C.c_float), C.POINTER(C.c_int)], None),
            "positions": ([C.c_float] * 4, None),
        }
        for name, (arguments, result) in signatures.items():
            target = getattr(self.dll, "reference_" + name)
            target.argtypes, target.restype = arguments, result

    def initialize(self, memory):
        self.errors = []
        self.null_card_reads = 0

        def read(address, target, size):
            if address == 0 and size == 4:
                # Match the failed Win32 read without changing its destination.
                self.null_card_reads += 1
                return 0
            try:
                C.memmove(target, memory.read(address, size), size)
                return 1
            except Exception as error:
                self.errors.append(error)
                return 0

        self.callback = MEMORY_READ(read)
        self.dll.reference_init(self.callback)

    def value(self, name):
        return self.dll.reference_global(name.encode("ascii"))

    def entity(self, player, obj, names):
        values = (C.c_double * 16)()
        self.dll.reference_entity(player, obj, values)
        return dict(zip(names, values, strict=True))

    def box(self, player, obj, attack, index):
        values = (C.c_int * 4)()
        self.dll.reference_box(player, obj, attack, index, values)
        return tuple(values)
