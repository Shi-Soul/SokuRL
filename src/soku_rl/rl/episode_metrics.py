"""Learner-relative episode summaries without mixing seats or missing measurements."""
from soku_rl.env.combat_metrics import summarize_combat


def summarize_episodes(records):
    counts = dict(win=0, loss=0, double_ko=0, time_limit=0)
    for record in records:
        outcome = record["outcome"]
        player = record["training_context"]["player"]
        if player not in (0, 1):
            raise ValueError("episode summaries require an actual learner seat")
        if outcome in ("p1_win", "p2_win"):
            outcome = "win" if outcome == f"p{player + 1}_win" else "loss"
        counts[outcome] += 1
    result = {"episodes": len(records), "counts": counts,
        "combat": summarize_combat([record.get("combat_metrics", {"available": False})
                                    for record in records])}
    if records:
        result.update(win_rate=counts["win"] / len(records),
            mean_decisions=sum(record["episode"]["l"] for record in records) / len(records),
            mean_frames=sum(record["frame"] for record in records) / len(records))
    return result


def grouped_episode_metrics(records):
    groups = {name: {} for name in ("learner_seat", "opponent", "opponent_character", "matchup")}
    for record in records:
        context = record["training_context"]
        player, opponent = context["player"], context["opponent"]
        keys = {"learner_seat": player, "opponent": opponent}
        if "match" in context:
            character = context["match"][f"player_{1 - player}"]["character"]
            keys.update(opponent_character=character, matchup=(player, opponent, character))
        for name, key in keys.items():
            groups[name].setdefault(key, []).append(record)
    return {"overall": summarize_episodes(records), "groups": {
        name: [{"key": key, **summarize_episodes(subset)} for key, subset in sorted(values.items())]
        for name, values in groups.items()}}
