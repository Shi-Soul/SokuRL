#include "LocalStart.hpp"
#include "ProfileDeck.hpp"
#include <BattleMode.hpp>
#include <BattleManager.hpp>
#include <Character.hpp>
#include <InputManager.hpp>
#include <Windows.h>
#include <cwchar>

namespace {
using ProfileInitializeMethod = void (__thiscall *)(SokuLib::Profile *, char);
constexpr DWORD P1_KEYMAP_MANAGER_PTR = 0x008989A0;
constexpr DWORD P2_KEYMAP_MANAGER_PTR = 0x0089918C;
constexpr DWORD P1_INPUT_MANAGER_PTR = 0x00898680;
constexpr DWORD P2_INPUT_MANAGER_PTR = 0x00898684;
constexpr DWORD P1_INPUT_DEVICE = 0x00898678;
constexpr DWORD INPUT_MANAGER_CLUSTER_DEVICE = 0x0089A2BC;
constexpr DWORD FALLBACK_KEY_MANAGER = 0x008986A8;
constexpr DWORD PROFILE_INITIALIZE = 0x00434BF0;

bool configureVsPlayer(SokuLib::PlayerInfo &info, bool right, std::uint32_t character,
    std::uint32_t palette, std::uint32_t deck)
{
    auto &profile = right ? SokuLib::profile2 : SokuLib::profile1;
    auto &source = SokuRLBridge::profileDeck(profile, character, deck);
    if (source.size != 20)
        return false;
    const auto initializeProfile = reinterpret_cast<ProfileInitializeMethod>(PROFILE_INITIALIZE);
    info.character = static_cast<SokuLib::Character>(character);
    info.isRight = right;
    info.palette = static_cast<unsigned char>(palette);
    info.deck = static_cast<unsigned char>(deck);
    info.effectiveDeck.clear();

    if (right) {
        *reinterpret_cast<SokuLib::KeyManager **>(P2_INPUT_MANAGER_PTR) =
            reinterpret_cast<SokuLib::KeyManager *>(FALLBACK_KEY_MANAGER);
        initializeProfile(&SokuLib::profile2, -1);
        info.keyManager = reinterpret_cast<SokuLib::KeyManager **>(P2_KEYMAP_MANAGER_PTR);
        for (int i = 0; i < source.size; ++i)
            info.effectiveDeck.push_back(source[i]);
        return true;
    }

    *reinterpret_cast<signed char *>(P1_INPUT_DEVICE) = -1;
    *reinterpret_cast<SokuLib::KeyManager **>(P1_INPUT_MANAGER_PTR) =
        reinterpret_cast<SokuLib::KeyManager *>(FALLBACK_KEY_MANAGER);
    initializeProfile(&SokuLib::profile1, -1);
    info.keyManager = reinterpret_cast<SokuLib::KeyManager **>(P1_KEYMAP_MANAGER_PTR);
    for (int i = 0; i < source.size; ++i)
        info.effectiveDeck.push_back(source[i]);
    return true;
}

}

namespace SokuRLBridge {
std::uint32_t environmentValue(const wchar_t *name, std::uint32_t fallback)
{
    wchar_t value[16]{};
    const auto length = GetEnvironmentVariableW(name, value, _countof(value));
    if (!length || length >= _countof(value))
        return fallback;
    wchar_t *end = nullptr;
    const auto parsed = wcstoul(value, &end, 10);
    return end && *end == L'\0' ? static_cast<std::uint32_t>(parsed) : fallback;
}

LocalStart readLocalStart()
{
    LocalStart config;
    config.p1Character = environmentValue(L"SOKURL_VS_P1_CHARACTER", 1);
    config.p2Character = environmentValue(L"SOKURL_VS_P2_CHARACTER", 0);
    config.p1Palette = environmentValue(L"SOKURL_VS_P1_PALETTE", 0);
    config.p2Palette = environmentValue(L"SOKURL_VS_P2_PALETTE", 0);
    config.p1Deck = environmentValue(L"SOKURL_VS_P1_DECK", 0);
    config.p2Deck = environmentValue(L"SOKURL_VS_P2_DECK", 0);
    config.stage = environmentValue(L"SOKURL_VS_STAGE", 0);
    config.music = environmentValue(L"SOKURL_VS_MUSIC", 0);
    config.pauseAtStart = environmentValue(L"SOKURL_VS_PAUSE_AT_START", 0) == 1;
    const auto seed = environmentValue(L"SOKURL_VS_SEED", 0xFFFFFFFFU);
    config.seedRequested = seed != 0xFFFFFFFFU;
    config.seed = seed;
    return config;
}

bool configureLocalStart(const LocalStart &config)
{
    *reinterpret_cast<signed char *>(INPUT_MANAGER_CLUSTER_DEVICE) = -1;
    SokuLib::setBattleMode(SokuLib::BATTLE_MODE_VSPLAYER,
        SokuLib::BATTLE_SUBMODE_PLAYING2);
    if (!configureVsPlayer(SokuLib::leftPlayerInfo, false, config.p1Character,
            config.p1Palette, config.p1Deck) ||
        !configureVsPlayer(SokuLib::rightPlayerInfo, true, config.p2Character,
            config.p2Palette, config.p2Deck)) {
        return false;
    }
    SokuLib::gameParams.stageId = static_cast<unsigned char>(config.stage);
    SokuLib::gameParams.musicId = static_cast<unsigned char>(config.music);
    if (config.seedRequested)
        SokuLib::gameParams.randomSeed = config.seed;
    return true;
}
}
