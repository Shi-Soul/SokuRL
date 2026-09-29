EXPORT void reference_init(MemoryRead reader) {
    memory_read = reader;
    my_data.~Character(); enemy_data.~Character();
    std::memset(&my_data,0,sizeof(my_data)); std::memset(&enemy_data,0,sizeof(enemy_data));
    new (&my_data) Character(Character::MY); new (&enemy_data) Character(Character::ENEMY);
    storage.values.clear();
}
EXPORT void reference_reload(unsigned int root, int mode, int active_weather) {
    weather = active_weather;
    my_data.SetRootAddress(root);
    enemy_data.SetRootAddress(root);
    my_data.Reload(static_cast<ObjBase::AI_MODE>(mode));
    enemy_data.Reload(static_cast<ObjBase::AI_MODE>(mode));
    is_bullethit();
}
EXPORT double reference_global(const char *name) { return storage.values.at(name); }
EXPORT void reference_entity(int player, int object, double *out) {
    const Character &c = player == 0 ? my_data : enemy_data;
    const ObjBase &p = object == -1 ? static_cast<const ObjBase &>(c) : c.GetObject(object);
    const double values[] = {p.x,p.y,p.speed.x,p.speed.y,double(p.dir),double(p.action),
        double(p.act_block),double(p.frame),double(p.hp),double(p.hit_stop),double(p.img_no),
        double(p.fflags),double(p.aflags),double(p.base_addr),
        double(p.hitarea.size()),double(p.attackarea.size())};
    std::copy(std::begin(values), std::end(values), out);
}
EXPORT void reference_box(int player, int object, int attack, int index, int *out) {
    const Character &c = player == 0 ? my_data : enemy_data;
    const ObjBase &p = object == -1 ? static_cast<const ObjBase &>(c) : c.GetObject(object);
    const Box &b = attack ? p.attackarea.at(index) : p.hitarea.at(index);
    out[0]=b.left; out[1]=b.top; out[2]=b.right; out[3]=b.bottom;
}
EXPORT int reference_field(int player, int kind, int index) {
    const Character &p = player == 0 ? my_data : enemy_data;
    switch (kind) {
    case 0: return GetCardId(player,index);
    case 1: return GetCardCost(player,index);
    case 2: return GetSkillLv(player,index);
    case 3: return p.sp_data[index];
    case 4: return p.GetKeyState(index);
    case 5: return int(p.object.size());
    }
    return -99999;
}
EXPORT void reference_keys(int operation, int value) {
    switch(operation) {
    case 0: key_on(value); break;
    case 1: key_off(value); break;
    case 2: key_reset(); break;
    case 3: keyboard.SetKeyDelay(value); break;
    case 4: keyboard.ProcessEvent(); break;
    case 5: keyboard=KeybdEvent(); std::memset(on,0,sizeof(on));
            std::memset(applied,0,sizeof(applied)); break;
    }
}
EXPORT int reference_key(int key, int requested) { return requested ? on[key] : applied[key]; }
EXPORT void reference_delayed_action(int delay, int action, int block, int frame) {
    enemy_data.action=action; enemy_data.act_block=block; enemy_data.frame=frame;
    reference_delay(delay);
}
EXPORT void reference_projectiles(float x, int count, const float *centres, const int *boxes) {
    my_data.x=x; enemy_data.object.clear();
    for(int i=0;i<count;i++) {
        Obj o(0,0); o.x=centres[i];
        Box b; b.left=boxes[2*i]; b.right=boxes[2*i+1]; b.top=b.bottom=0;
        o.attackarea.push_back(b); enemy_data.object.push_back(o);
    }
    is_bullethit();
    storage.values["obj_dis"]=obj_dis; storage.values["obj_dis2"]=obj_dis2;
}
