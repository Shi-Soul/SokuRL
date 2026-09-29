"""Build the upstream C++ reader with callbacks in place of operating-system I/O."""
import re
from pathlib import Path


def function(source, signature):
    start = source.index(signature)
    end = source.index(b"\n}", start) + 2
    return source[start:end]


def generate(upstream, output):
    upstream, output = Path(upstream), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    addresses = re.findall(rb"SWRS_ADDR_(\w+)\s*=\s*(0x[0-9A-Fa-f]+)",
                           (upstream / "SWRSAddr.ini").read_bytes())
    parts = [b"#include \"shim.hpp\"\n"]
    parts.extend(b"#define ADDR_" + name + b" " + value + b"\n" for name, value in addresses)
    # The shipped INI omits this field; address.h supplies its 1.10a value.
    default = re.search(rb"set_default\(ACTIONBLOCKIDOFS,\s*0x[0-9A-Fa-f]+,\s*(0x[0-9A-Fa-f]+)",
                        (upstream / "address.h").read_bytes()).group(1)
    parts.append(b"#define ADDR_ACTIONBLOCKIDOFS " + default + b"\n")
    utility = (upstream / "utility.h").read_bytes()
    parts.append(utility[:utility.index(b"bool IsSWR")])
    for name in ("obj_base.h", "obj.h", "character.h"):
        source = (upstream / name).read_bytes()
        if name == "obj.h":
            # ObjBase's float read filter can reject negative zero or tiny
            # speeds. The published Obj constructor leaves the destination
            # uninitialized. Complete that missing initialization explicitly;
            # otherwise the reference exposes unrelated host stack bytes.
            source = source.replace(b"base_addr = addr;", b"base_addr = addr; x = y = speed.x = speed.y = 0;")
        parts.append(source)
    parts.append(b"Character my_data(Character::MY), enemy_data(Character::ENEMY);\n")
    parts.append(b"short GetCardId(int player,int n);\n")
    for name in ("obj_base.cpp", "character.cpp"):
        source = (upstream / name).read_bytes().replace(b'#include "stdafx.h"', b"")
        if name == "obj_base.cpp":
            # The packaged 0.93 machine code performs x87 addition before
            # integer conversion, and transforms hit boxes when flag != 0.
            source = source.replace(b"static_cast<int>(x)", b"static_cast<double>(x)")
            source = source.replace(b"static_cast<int>(y)", b"static_cast<double>(y)")
            marker = b"ReadProcessMemory(ph,  base_addr + ADDR_HITAREA2OFS + i * sizeof(box), &box, sizeof(box));"
            source = source.replace(marker, marker + b"\nif(flag) {"
                b"box.left = x + static_cast<double>(box.left); box.right = x + static_cast<double>(box.right);"
                b"box.top = -y + static_cast<double>(box.top); box.bottom = -y + static_cast<double>(box.bottom); }\n")
        if name == "character.cpp":
            source = source.replace(b"object[i].img_no!=0x1B3", b"object[i].img_no!=0x1B4")
            source = source.replace(b"361 - object[i].frame", b"601 - object[i].frame")
            for field in (b"DROPWATERTIMEOFS", b"RYUUSEIOFS"):
                marker = b"ReadProcessMemory(ph, base_addr+ADDR_" + field + b", mini);\n\t\t\tdata = mini;"
                source = source.replace(marker, marker.replace(b"data = mini;", b"data = static_cast<unsigned short>(mini);"))
            # MSVC accepts false as a null pointer constant; GCC requires 0.
            signature = b"const Obj *Character::GetOptionObject("
            before, after = source.split(signature)
            source = before + signature + after.replace(b"return false;", b"return 0;")
        parts.append(source)
    main = (upstream / "main.cpp").read_bytes()
    for signature in (b"int GetPlayerAddr(", b"int is_CardUse(", b"short GetCardId(", b"int GetCardCost(", b"int GetCardCost2(",
                      b"char GetSkillLv(", b"int GetSpecialData(", b"int get_correction(",
                      b"void is_bullethit("):
        body = function(main, signature)
        if signature == b"void is_bullethit(":
            body = body.replace(b"BOOST_FOREACH(const Box &box, obj.attackarea)",
                                b"if(obj.attackarea.size() >= 16) continue;\nBOOST_FOREACH(const Box &box, obj.attackarea)")
        parts.append(body)
    keys = (upstream / "key_manager.cpp").read_bytes()
    parts.append((upstream / "keybd_event.h").read_bytes())
    parts.append(b"KeybdEvent keyboard; int key[10]={0,1,2,3,4,5,6,7,8,9}; char on[10]={};")
    parts.append(b"void key_on(int n); void key_off(int n);\n")
    for signature in (b"char get_key_stat(", b"void on_check(", b"void key_on(",
                      b"void key_off(", b"void key_reset("):
        parts.append(function(keys, signature))
    events = (upstream / "keybd_event.cpp").read_bytes()
    parts.append(function(events, b"void KeybdEvent::AddEvent("))
    parts.append(function(events, b"void KeybdEvent::ProcessEvent("))
    parts.append(b"void KeybdEvent::ExecEvent(BYTE code,DWORD flags){ applied[code]=flags==0; }\n")
    parts.append(b"void reference_delay(int get_delay) {\n"
                 b"static short *act=NULL,*act_block=NULL; static int *frame=NULL; static int delay=0;\n")
    start = main.index(b"\tif(get_delay != delay)")
    end = main.index(b"\n\twhile(1)", start)
    parts.append(main[start:end])
    start = main.index(b'\tif(delay == 0)', end)
    end = main.index(b'\n\tengine->setScriptValueBool("is_th105"', start)
    parts.append(main[start:end] + b"\n}\n")
    start = main.index(b"\t\t\tdis = abs(my_data.x-enemy_data.x);")
    end = main.index(b"\n\t\t\tbreak;", start)
    parts.append(b"void reference_distance(){ int dis, dis_y, dis2;\n" + main[start:end]
                 + b'\nengine->setScriptValue("dis",dis); engine->setScriptValue("dis_y",dis_y);'
                   b'engine->setScriptValue("dis2",dis2); }\n')
    parts.append((Path(__file__).parent / "exports.cpp").read_bytes())
    (output / "reference.cpp").write_bytes(b"\n".join(parts))
    (output / "shim.hpp").write_bytes((Path(__file__).parent / "shim.hpp").read_bytes())
    (output / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\nproject(GodReference LANGUAGES CXX)\n"
        "add_library(god_reference SHARED reference.cpp)\n"
        "target_compile_features(god_reference PRIVATE cxx_std_17)\n"
        "target_compile_definitions(god_reference PRIVATE _CRT_SECURE_NO_WARNINGS)\n",
        encoding="utf-8")
