#include "FrameState.hpp"
#include <BattleManager.hpp>
#include <BattleMode.hpp>
#include <Character.hpp>
#include <SokuAddresses.hpp>
#include <Weather.hpp>
#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>

namespace {
void applySimplePlayerState(SokuLib::CharacterManager &manager,
    const SokuRLBridge::SimplePlayerState &state)
{
    manager.objectBase.position.x = state.x;
    manager.objectBase.position.y = state.y;
    manager.objectBase.speed.x = state.speedX;
    manager.objectBase.speed.y = state.speedY;
    manager.objectBase.direction = static_cast<SokuLib::Direction>(state.facing);
    manager.objectBase.hp = static_cast<short>(state.hp);
    manager.currentSpirit = static_cast<unsigned short>(static_cast<std::int16_t>(state.spirit));
    manager.maxSpirit = static_cast<unsigned short>(static_cast<std::int16_t>(state.maxSpirit));
    manager.cardGauge = static_cast<unsigned short>(state.cardGauge);
    manager.cardCount = static_cast<unsigned char>(state.cardCount);
}

constexpr std::uint64_t FNV_OFFSET = 14695981039346656037ULL;
constexpr std::uint64_t FNV_PRIME = 1099511628211ULL;

std::uint64_t hashBytes(std::uint64_t hash, const void *data, std::size_t size)
{
    const auto *bytes = static_cast<const unsigned char *>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

void captureObject(const SokuLib::ObjectManager &object, std::uint32_t owner,
    std::uint32_t index, SokuRLBridge::ObjectState &state)
{
    const auto &projectile = reinterpret_cast<const SokuLib::ProjectileManager &>(object);
    state.ownerIndex = owner;
    state.listIndex = index;
    state.typeId = reinterpret_cast<std::uint32_t>(object.vtable);
    state.actionId = static_cast<std::uint32_t>(object.action);
    state.actionBlockId = object.actionBlockId;
    state.animationCounter = object.animationCounter;
    state.animationSubFrame = object.animationSubFrame;
    state.frameCount = object.frameCount;
    state.x = object.position.x;
    state.y = object.position.y;
    state.speedX = object.speed.x;
    state.speedY = object.speed.y;
    // Several projectile classes leave this inherited slot uninitialized.
    // Position and speed are stable; expose a canonical value instead of heap noise.
    state.gravity = 0.0f;
    state.direction = object.direction;
    state.hp = object.hp;
    state.hitstop = object.hitstop;
    state.hitBoxCount = object.hitBoxCount;
    state.hurtBoxCount = object.hurtBoxCount;
    state.characterIndex = projectile.characterIndex;
    state.isActive = projectile.isActive;
}

void captureObjects(const SokuLib::CharacterManager &manager, std::uint32_t owner,
    SokuRLBridge::ObjectState (&states)[SokuRLBridge::MAX_OBJECTS_PER_PLAYER],
    std::uint32_t &count, std::uint32_t &overflow)
{
    const auto &list = manager.objects.list;
    overflow = list.size > SokuRLBridge::MAX_OBJECTS_PER_PLAYER ? 1U : 0U;
    if (!list.head || !list.size)
        return;

    auto *node = list.head->next;
    while (node && node != list.head && count < SokuRLBridge::MAX_OBJECTS_PER_PLAYER) {
        if (node->val)
            captureObject(*node->val, owner, count, states[count]);
        else
            overflow = 1;
        ++count;
        node = node->next;
    }
    if (count < std::min<std::uint32_t>(list.size, SokuRLBridge::MAX_OBJECTS_PER_PLAYER))
        overflow = 1;
}

void captureHand(const SokuLib::CharacterManager &manager, SokuRLBridge::PlayerState &state)
{
    std::fill(std::begin(state.handIds), std::end(state.handIds), -1);
    const auto count = std::min<unsigned>(manager.cardCount, 5U);
    if (!manager.hand.handCardBase || manager.hand.handCardMax <= 0)
        return;
    for (unsigned i = 0; i < count; ++i) {
        const auto index = (manager.hand.selectedCard + static_cast<int>(i)) % manager.hand.handCardMax;
        const auto *card = manager.hand.handCardBase[index < 0 ? index + manager.hand.handCardMax : index];
        if (card)
            state.handIds[i] = card->id;
    }
}

void capturePlayer(const SokuLib::CharacterManager &manager, const SokuRLBridge::LogicalInput &input,
    SokuRLBridge::PlayerState &state)
{
    std::memset(&state, 0, sizeof(state));
    state.characterId = manager.characterIndex;
    state.x = manager.objectBase.position.x;
    state.y = manager.objectBase.position.y;
    state.speedX = manager.objectBase.speed.x;
    state.speedY = manager.objectBase.speed.y;
    state.facing = manager.objectBase.direction;
    state.hp = manager.objectBase.hp;
    state.spirit = static_cast<std::int16_t>(manager.currentSpirit);
    state.maxSpirit = static_cast<std::int16_t>(manager.maxSpirit);
    state.cardGauge = manager.cardGauge;
    state.cardCount = manager.cardCount;
    captureHand(manager, state);
    state.actionId = static_cast<std::uint32_t>(manager.objectBase.action);
    state.sequenceId = manager.objectBase.actionBlockId;
    state.subsequenceId = manager.objectBase.animationCounter;
    state.animationFrame = manager.objectBase.frameData ? manager.objectBase.frameData->number : 0;
    state.elapsedInSubsequence = manager.objectBase.frameCount;
    state.hitstop = manager.objectBase.hitstop;
    state.untech = manager.untech;
    if (manager.objectBase.frameData) {
        state.airborne = manager.objectBase.frameData->frameFlags.airborne ? 1U : 0U;
        state.frameFlags = manager.objectBase.frameData->frameFlags.value;
        state.attackFlags = manager.objectBase.frameData->attackFlags.value;
    }
    state.objectCount = manager.objects.list.size;
    state.input = input;
}

}

namespace SokuRLBridge {
SokuRLBridge::SimpleStatePatch simplePatchFrom(const SokuRLBridge::RawFrameState &state)
{
    SokuRLBridge::SimpleStatePatch patch{};
    patch.timeElapsedRaw = state.timeElapsedRaw;
    patch.activeWeather = state.activeWeather;
    patch.displayedWeather = state.displayedWeather;
    patch.weatherCounter = state.weatherCounter;
    const auto copyPlayer = [](const SokuRLBridge::PlayerState &source,
        SokuRLBridge::SimplePlayerState &target) {
        target.x = source.x;
        target.y = source.y;
        target.speedX = source.speedX;
        target.speedY = source.speedY;
        target.facing = source.facing;
        target.hp = source.hp;
        target.spirit = source.spirit;
        target.maxSpirit = source.maxSpirit;
        target.cardGauge = source.cardGauge;
        target.cardCount = source.cardCount;
    };
    copyPlayer(state.p1, patch.p1);
    copyPlayer(state.p2, patch.p2);
    return patch;
}

bool isValidSimplePlayerState(const SokuRLBridge::SimplePlayerState &state)
{
    constexpr auto minimum = std::numeric_limits<std::int16_t>::min();
    constexpr auto maximum = std::numeric_limits<std::int16_t>::max();
    return state.spirit >= minimum && state.spirit <= maximum &&
        state.maxSpirit >= minimum && state.maxSpirit <= maximum;
}

void applySimpleState(SokuLib::BattleManager &manager,
    const SokuRLBridge::SimpleStatePatch &state)
{
    *reinterpret_cast<std::uint32_t *>(SokuLib::ADDR_TIME_ELAPSED) = state.timeElapsedRaw;
    SokuLib::activeWeather = static_cast<SokuLib::Weather>(state.activeWeather);
    SokuLib::displayedWeather = static_cast<SokuLib::Weather>(state.displayedWeather);
    SokuLib::weatherCounter = static_cast<unsigned short>(state.weatherCounter);
    applySimplePlayerState(manager.leftCharacterManager, state.p1);
    applySimplePlayerState(manager.rightCharacterManager, state.p2);
}

std::uint64_t stateHash(const SokuRLBridge::RawFrameState &state)
{
    std::uint64_t hash = FNV_OFFSET;
    hash = hashBytes(hash, &state.frameId, sizeof(state.frameId));
    hash = hashBytes(hash, &state.sceneId, sizeof(state.sceneId));
    hash = hashBytes(hash, &state.battleMode, sizeof(state.battleMode));
    hash = hashBytes(hash, &state.battleSubMode, sizeof(state.battleSubMode));
    hash = hashBytes(hash, &state.stageId, sizeof(state.stageId));
    hash = hashBytes(hash, &state.roundId, sizeof(state.roundId));
    hash = hashBytes(hash, &state.timeElapsedRaw, sizeof(state.timeElapsedRaw));
    hash = hashBytes(hash, &state.activeWeather, sizeof(state.activeWeather));
    hash = hashBytes(hash, &state.displayedWeather, sizeof(state.displayedWeather));
    hash = hashBytes(hash, &state.weatherCounter, sizeof(state.weatherCounter));
    hash = hashBytes(hash, &state.randomSeed, sizeof(state.randomSeed));
    hash = hashBytes(hash, &state.p1, sizeof(state.p1));
    hash = hashBytes(hash, &state.p2, sizeof(state.p2));
    hash = hashBytes(hash, &state.p1ObjectCount, sizeof(state.p1ObjectCount));
    hash = hashBytes(hash, &state.p2ObjectCount, sizeof(state.p2ObjectCount));
    hash = hashBytes(hash, &state.p1ObjectOverflow, sizeof(state.p1ObjectOverflow));
    hash = hashBytes(hash, &state.p2ObjectOverflow, sizeof(state.p2ObjectOverflow));
    hash = hashBytes(hash, state.p1Objects, sizeof(state.p1Objects));
    return hashBytes(hash, state.p2Objects, sizeof(state.p2Objects));
}

SokuRLBridge::RawFrameState captureState(SokuLib::BattleManager *manager, std::uint64_t frame,
    std::uint32_t segment, const LogicalInput (&inputs)[2])
{
    SokuRLBridge::RawFrameState state{};
    state.frameId = frame;
    state.segmentId = segment;
    state.sceneId = *reinterpret_cast<const std::uint32_t *>(SokuLib::ADDR_SCENE_ID);
    state.battleMode = static_cast<std::uint32_t>(SokuLib::mainMode);
    state.battleSubMode = static_cast<std::uint32_t>(SokuLib::subMode);
    state.stageId = SokuLib::gameParams.stageId;
    state.roundId = static_cast<unsigned char>(manager->currentRound);
    state.timeElapsedRaw = *reinterpret_cast<const std::uint32_t *>(SokuLib::ADDR_TIME_ELAPSED);
    state.activeWeather = static_cast<std::uint32_t>(SokuLib::activeWeather);
    state.displayedWeather = static_cast<std::uint32_t>(SokuLib::displayedWeather);
    state.weatherCounter = SokuLib::weatherCounter;
    state.randomSeed = SokuLib::gameParams.randomSeed;
    capturePlayer(manager->leftCharacterManager, inputs[0], state.p1);
    capturePlayer(manager->rightCharacterManager, inputs[1], state.p2);
    state.p1.characterId = static_cast<std::uint32_t>(SokuLib::gameParams.leftPlayerInfo.character);
    state.p2.characterId = static_cast<std::uint32_t>(SokuLib::gameParams.rightPlayerInfo.character);
    captureObjects(manager->leftCharacterManager, 0, state.p1Objects,
        state.p1ObjectCount, state.p1ObjectOverflow);
    captureObjects(manager->rightCharacterManager, 1, state.p2Objects,
        state.p2ObjectCount, state.p2ObjectOverflow);
    state.stateHash = stateHash(state);
    return state;
}

}
