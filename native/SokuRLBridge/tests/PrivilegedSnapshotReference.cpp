#include "../PrivilegedSnapshot.hpp"
#if defined(_WIN32)
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif
EXPORT void captureSnapshot(SokuRLBridge::PrivilegedSnapshot *snapshot, SokuRLBridge::SnapshotRead read) {
    SokuRLBridge::capturePrivilegedSnapshot(*snapshot, read);
}
