#include "NetworkStart.hpp"

#include <Scenes.hpp>
#include <Menus.hpp>
#include <Windows.h>
#include <cstring>

namespace
{
enum class Role { Disabled, Host, Join };
Role g_role = Role::Disabled;
unsigned g_port = 0;
char g_address[16]{};
bool g_started = false;

bool readRequired(const char *name, char *buffer, DWORD capacity)
{
    const auto length = GetEnvironmentVariableA(name, buffer, capacity);
    return length > 0 && length < capacity;
}

bool parseDecimal(const char *text, unsigned maximum, unsigned &value)
{
    if (!*text)
        return false;
    value = 0;
    for (; *text; ++text) {
        if (*text < '0' || *text > '9')
            return false;
        const auto digit = static_cast<unsigned>(*text - '0');
        if (value > maximum / 10 ||
            (value == maximum / 10 && digit > maximum % 10))
            return false;
        value = value * 10 + digit;
    }
    return true;
}

bool isIPv4(const char *address)
{
    unsigned groups = 0;
    const char *start = address;
    for (const char *cursor = address;; ++cursor) {
        if (*cursor != '.' && *cursor != '\0')
            continue;
        const auto length = cursor - start;
        if (length < 1 || length > 3)
            return false;
        char component[4]{};
        std::memcpy(component, start, length);
        unsigned value = 0;
        if (!parseDecimal(component, 255, value))
            return false;
        ++groups;
        if (!*cursor)
            return groups == 4;
        if (groups >= 4)
            return false;
        start = cursor + 1;
    }
}
}

namespace SokuRLBridge
{
bool initializeNetworkStart(bool offlineBootstrap, bool unlimitedPacing)
{
    char role[8]{};
    SetLastError(ERROR_SUCCESS);
    const auto length = GetEnvironmentVariableA("SOKURL_NETWORK_ROLE", role, sizeof(role));
    if (!length && GetLastError() == ERROR_ENVVAR_NOT_FOUND)
        return true;
    if (!length || length >= sizeof(role) || offlineBootstrap || unlimitedPacing)
        return false;
    if (std::strcmp(role, "host") == 0)
        g_role = Role::Host;
    else if (std::strcmp(role, "join") == 0)
        g_role = Role::Join;
    else
        return false;

    char port[6]{};
    if (!readRequired("SOKURL_NETWORK_PORT", port, sizeof(port)) ||
        !parseDecimal(port, 65535, g_port) || !g_port)
        return false;
    if (g_role == Role::Join &&
        (!readRequired("SOKURL_NETWORK_ADDRESS", g_address, sizeof(g_address)) ||
            !isIPv4(g_address)))
        return false;
    return true;
}

void processNetworkStart(const SokuLib::Title &title, int nextScene)
{
    if (g_role == Role::Disabled || g_started || nextScene != SokuLib::SCENE_TITLE)
        return;
    // th123 1.10a Title::onProcess (0x4278DD, 0x42791C) uses these
    // same conditions before accepting a menu selection. Do not block its thread.
    const auto menuFrames = *reinterpret_cast<const int *>(
        reinterpret_cast<const unsigned char *>(&title) + 0x68C);
    if (title.menuState == 0 || menuFrames < 16 || SokuLib::menuManager.isInMenu)
        return;
    // Automated menu entry has no physical key event to select its input device.
    // setBattleMode reads this signed byte through 0x40A8E0; -1 selects keyboard.
    // Leaving zero selects joystick 0 even when the device list is empty.
    *reinterpret_cast<signed char *>(0x0089A2BC) = -1;
    auto *menu = SokuLib::MenuConnect::create();
    SokuLib::activateMenu(menu);
    g_started = true;
    if (g_role == Role::Host)
        menu->setupHost(g_port, false);
    else
        menu->joinHost(g_address, g_port, false);
}
}
