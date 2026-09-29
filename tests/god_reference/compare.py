"""Compare every exposed fighter field and object with the upstream reader."""
from soku_rl.env.observation.memory_schema import OBJECT_NAMES, FIGHTER_FIELDS


def players_equal(reference, players):
    for seat, player in enumerate(players):
        prefix = "my_" if seat == 0 else "enemy_"
        for name in (*FIGHTER_FIELDS, "char", "spell", "card", "obj_n"):
            assert player[name] == reference.value(prefix + name), (seat, name, player[name], reference.value(prefix+name))
        for index, entity in enumerate((player, *player["objects"])):
            expected = reference.entity(seat, index - 1, OBJECT_NAMES)
            for name, value in expected.items():
                assert entity[name] == value, (seat, index, name, entity[name], value)
            for attack, kind in enumerate(("hitarea", "attackarea")):
                for box_index, box in enumerate(entity[kind]):
                    assert box == reference.box(seat, index - 1, attack, box_index), (seat,index,kind,box_index)
        for kind, name in ((2, "skills"), (3, "special"), (4, "keys")):
            limit = 15 if name == "skills" else len(player[name])
            for index in range(limit):
                assert player[name][index] == reference.dll.reference_field(seat, kind, index), (seat,name,index)
        for index in range(5):
            assert player["cards"][2*index] == reference.dll.reference_field(seat, 0, index)
            assert player["cards"][2*index+1] == reference.dll.reference_field(seat, 1, index)
    assert not reference.errors
