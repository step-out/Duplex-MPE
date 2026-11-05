"""Read the public dataset and validate portable audio references."""

import hashlib
import json
from collections import Counter
from pathlib import Path

EVENTS = {"T", "N1", "N2", "N3", "N4_Q", "N4_R"}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def resolve_audio(root, path):
    root, relative = Path(root).resolve(), Path(path)
    resolved = (root / relative).resolve()
    if relative.is_absolute() or not resolved.is_relative_to(root):
        raise ValueError("Audio path must stay within the dataset directory")
    return resolved


def load(root, condition="all"):
    if condition not in {"all", "explicit", "implicit"}:
        raise ValueError("Unknown addressing condition")
    scenes = []
    for side in ("explicit", "implicit") if condition == "all" else (condition,):
        group = load_json(Path(root) / "manifests" / f"{side}.json")
        for scene in group:
            if scene["addressing"] != side:
                raise ValueError("Manifest addressing mismatch")
        scenes.extend(group)
    if len({scene["scenario_id"] for scene in scenes}) != len(scenes):
        raise ValueError("Duplicate scenario ID")
    return scenes


def validate(root, full_audio=False):
    root = Path(root)
    scenes = load(root)
    inventory = load_json(root / "audio_inventory.json")
    assets = {item["path"]: item for item in inventory}
    if len(assets) != len(inventory):
        raise ValueError("Duplicate audio inventory path")
    pairs, counts = {}, Counter()
    for scene in scenes:
        sid, turns = scene["scenario_id"], scene["turns"]
        pair = pairs.setdefault(scene["pair_id"], {})
        if scene["addressing"] in pair:
            raise ValueError("Duplicate condition within pair: " + sid)
        pair[scene["addressing"]] = scene
        requests = [i for i, turn in enumerate(turns) if turn["event_type"] == "T"]
        if requests != [scene["trigger_index"]]:
            raise ValueError("Expected exactly one T: " + sid)
        if not scene["gold_answer"].strip():
            raise ValueError("Empty gold answer: " + sid)
        if any(i < 0 or i >= requests[0] for i in scene["supporting_indices"]):
            raise ValueError("Non-causal supporting index: " + sid)
        for index, turn in enumerate(turns):
            event = turn["event_type"]
            if event not in EVENTS or turn["turn_index"] != index:
                raise ValueError("Invalid turn: " + sid)
            if event == "N4_Q" and (
                index + 1 == len(turns) or turns[index + 1]["event_type"] != "N4_R"
            ):
                raise ValueError("Unpaired N4 question: " + sid)
            if event == "N4_R" and (
                index == 0 or turns[index - 1]["event_type"] != "N4_Q"
            ):
                raise ValueError("Unpaired N4 resolution: " + sid)
            if turn["audio_path"] not in assets:
                raise ValueError("Missing audio inventory entry: " + sid)
            counts[scene["addressing"] + ":" + event] += 1
    for pair_id, pair in pairs.items():
        if set(pair) != {"explicit", "implicit"}:
            raise ValueError("Missing paired condition: " + pair_id)
        if pair["explicit"]["gold_answer"] != pair["implicit"]["gold_answer"]:
            raise ValueError("Different paired gold answers: " + pair_id)
    for item in inventory:
        path = resolve_audio(root, item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError("Missing or truncated audio: " + item["path"])
        if full_audio:
            import soundfile as sf

            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Audio checksum mismatch: " + item["path"])
            info = sf.info(path)
            if (info.samplerate, info.frames, info.channels) != (
                item["sample_rate"],
                item["frames"],
                item["channels"],
            ):
                raise ValueError("Audio header mismatch: " + item["path"])
    return {
        "scenarios": len(scenes),
        "pairs": len(pairs),
        "audio_files": len(inventory),
        "turn_counts": dict(counts),
        "all_audio_checksums_verified": full_audio,
    }
