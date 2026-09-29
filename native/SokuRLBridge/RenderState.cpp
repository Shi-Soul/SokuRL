#include "RenderState.hpp"
#include <BattleManager.hpp>
#include <Camera.hpp>
#include <Weather.hpp>
#include <cstring>

namespace SokuRLBridge {
namespace {
void entity(const SokuLib::ObjectManager &object, RenderEntity &state)
{
    state.x = object.position.x;
    state.y = object.position.y;
    state.alpha = object.renderInfos.color.a / 255.0f;
    state.facing = object.direction;
    // This flag describes available render data, not a proof of final visibility.
    state.drawable = object.image && object.frameData ? 1U : 0U;
}

void objects(const SokuLib::CharacterManager &player, RenderState &state, unsigned owner)
{
    const auto &list = player.objects.list;
    if (!list.size) return;
    if (!list.head) { state.overflow = 1; return; }
    auto *node = list.head->next;
    unsigned visited = 0;
    while (node && node != list.head && visited < list.size) {
        if (!node->val) {
            state.overflow = 1;
            break;
        }
        RenderEntity object{};
        entity(*node->val, object);
        captureRenderObject(state, object, owner);
        ++visited;
        node = node->next;
    }
    if (visited != list.size || node != list.head)
        state.overflow = 1;
}
}

void captureRenderState(RenderState &state)
{
    std::memset(&state, 0, sizeof(state));
    state.cameraX = SokuLib::camera.translate.x;
    state.cameraY = SokuLib::camera.translate.y;
    state.cameraScale = SokuLib::camera.scale;
    state.weather = static_cast<std::uint32_t>(SokuLib::activeWeather);
    auto &battle = SokuLib::getBattleMgr();
    entity(battle.leftCharacterManager.objectBase, state.players[0]);
    entity(battle.rightCharacterManager.objectBase, state.players[1]);
    objects(battle.leftCharacterManager, state, 0);
    objects(battle.rightCharacterManager, state, 1);
}
}
