#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <string>
#include <vector>

#define BOOST_ASSERT assert
#define BOOST_FOREACH(declaration, collection) for (declaration : collection)
#define KEYEVENTF_KEYUP 2
using BYTE = unsigned char;
using DWORD = unsigned int;
namespace org { namespace click3 { namespace Utility { using SHARED_HANDLE = int; } } }
enum {ACT_UP,ACT_DOWN,ACT_LEFT,ACT_RIGHT,ACT_A,ACT_B,ACT_C,ACT_D,ACT_AB,ACT_BC,
      ACT_DLEFT,ACT_DRIGHT,ACT_ULEFT,ACT_URIGHT};
enum {MY, ENEMY};
using MemoryRead = int (*)(unsigned int, void *, unsigned int);
MemoryRead memory_read;
int weather = 0, ph = 0, obj_dis = 0, obj_dis2 = 0;
int applied[10] = {};
bool IsSWR() { return false; }
struct Engine {
    std::map<std::string, double> values;
    void setScriptValue(const char *name, double value) { values[name] = value; }
} storage;
Engine *engine = &storage;
bool ReadProcessMemory(int, unsigned int address, void *buffer, unsigned int size) {
    return memory_read(address, buffer, size) != 0;
}
bool ReadProcessMemory(int handle, unsigned int address, void *buffer, unsigned int size,
                       unsigned int *) {
    return ReadProcessMemory(handle, address, buffer, size);
}
template<class T> bool ReadProcessMemory(int handle, unsigned int address, T &value) {
    return ReadProcessMemory(handle, address, &value, sizeof(value));
}
// lib.cpp rejects NaN, infinities, negative zero, and nonzero floats whose
// exponent is below 108. Keep the old field value in those cases.
bool ReadProcessMemory(int handle, unsigned int address, float &value) {
    unsigned int bits;
    if (!ReadProcessMemory(handle, address, bits)) return false;
    const auto exponent = (bits >> 23) & 255;
    if (exponent > 254 || (bits != 0 && exponent < 108)) return false;
    std::memcpy(&value, &bits, 4);
    return true;
}
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif
