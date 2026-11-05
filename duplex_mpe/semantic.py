"""Scored ASR windows and offline semantic judge requests and verdicts."""

import hashlib
import json
import math
from pathlib import Path
import re

from .dataset import save_json
from .metrics import collect_m2_cases, collect_non_t_episodes
from .prompts import JUDGE_PROMPT, SYS, TMPL


def tasks_for(reports, scenes):
    """Build voiced SP episodes and answered AA windows from timing reports."""
    tasks = [
        {**episode, "kind": "silence"}
        for episode in collect_non_t_episodes(reports, scenes)
    ]
    for case in collect_m2_cases(reports):
        scene = scenes[case["scenario_id"]]
        index = case["turn_index"]
        context = "\n".join(
            f"{turn.get('speaker', '?')}: {turn.get('text', '')}"
            for turn in scene["turns"][:index]
            if turn.get("text")
        )
        tasks.append(
            {
                **case,
                "kind": "answer",
                "context": context or "(none)",
                "request": scene["turns"][index]["text"],
                "gold": scene["gold_answer"],
            }
        )
    for task in tasks:
        audio_hash = reports[task["scenario_id"]].get("assistant_audio_sha256")
        if audio_hash is not None:
            task["expected_audio_sha256"] = audio_hash
    return sorted(
        tasks, key=lambda task: (task["scenario_id"], task["turn_index"], task["kind"])
    )


def _task_key(task):
    if task["kind"] not in ("answer", "silence"):
        raise ValueError("Unknown semantic task kind: " + str(task["kind"]))
    return task["scenario_id"] + ":" + str(task["turn_index"]) + ":" + task["kind"]


def _audio_hash(task):
    actual = task.get("assistant_audio_sha256")
    expected = task.get("expected_audio_sha256")
    if actual is not None and expected is not None and actual != expected:
        raise ValueError("Conflicting task audio hashes: " + _task_key(task))
    return actual if actual is not None else expected


def _bind_audio(label, task):
    expected = _audio_hash(task)
    stored = label.get("assistant_audio_sha256")
    if expected is not None:
        if stored is not None and stored != expected:
            raise ValueError("Label audio hash differs from task: " + _task_key(task))
        return {**label, "assistant_audio_sha256": expected}
    return label


def _index_tasks(tasks, require_hyp=True):
    indexed = {}
    for task in tasks:
        key = _task_key(task)
        if key in indexed:
            raise ValueError("Duplicate task " + key)
        if require_hyp and not isinstance(task.get("hyp"), str):
            raise ValueError("Missing ASR transcript " + key)
        _audio_hash(task)
        indexed[key] = task
    return indexed


def _validate_segments(task, wave_duration):
    segments = task.get("segments")
    if not isinstance(segments, (list, tuple)) or not segments:
        raise ValueError("ASR task requires nonempty segments: " + _task_key(task))
    previous_end = 0.0
    for segment in segments:
        if not isinstance(segment, (list, tuple)) or len(segment) != 2:
            raise ValueError("Invalid ASR segment pair: " + _task_key(task))
        start, end = segment
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in (start, end)
        ):
            raise ValueError("ASR segment times must be finite: " + _task_key(task))
        if not 0 <= start < end <= wave_duration + 0.04 or start < previous_end:
            raise ValueError(
                "ASR segments must be ordered, nonoverlapping, and within audio: "
                + _task_key(task)
            )
        previous_end = end


def transcribe(
    tasks, audio_dir, out, model_dir="Qwen/Qwen3-ASR-1.7B", device="cuda", batch_size=16
):
    """Concatenate detected segments per window and run English Qwen3-ASR.

    Clips shorter than 0.15 seconds have an empty transcript. Cached windows
    are reused only when their task, source audio, and ASR settings match.
    Segment ends may exceed audio by up to 40 ms for report frame rounding.
    The optional ASR model is loaded only if transcription is needed.
    """
    import numpy as np
    import soundfile as sf

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("ASR batch_size must be a positive integer")
    indexed = _index_tasks(tasks, require_hyp=False)
    previous = {}
    if Path(out).exists():
        previous = _index_tasks(json.loads(Path(out).read_text()))
    checkpoint = {key: value for key, value in previous.items() if key in indexed}
    by_scene = {}
    for task in indexed.values():
        by_scene.setdefault(task["scenario_id"], []).append(task)
    output = []
    model = None
    for sid, group in by_scene.items():
        scene_start = len(output)
        scene_changed = False
        path = Path(audio_dir) / (sid + ".wav")
        if not path.exists():
            path = Path(audio_dir) / (sid + ".flac")
        audio_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        for task in group:
            expected = task.get("expected_audio_sha256")
            if expected is not None and expected != audio_hash:
                raise ValueError("Assistant audio differs from timing report: " + sid)
        wave, sr = sf.read(path)
        if wave.ndim != 1:
            raise ValueError("Assistant waveform must be mono")
        batch, pending = [], []
        for task in group:
            _validate_segments(task, len(wave) / sr)
            record = {
                **task,
                "asr_model": model_dir,
                "asr_max_new_tokens": 2048,
                "asr_device": device,
                "asr_dtype": "float16" if device != "cpu" else "default",
                "assistant_audio_sha256": audio_hash,
            }
            old = previous.get(_task_key(task))
            # JSON normalizes segment tuples to lists when writing the cache.
            normalized = json.loads(json.dumps(record))
            if old and all(old.get(key) == value for key, value in normalized.items()):
                output.append(old)
                continue
            scene_changed = True
            clip = np.concatenate(
                [
                    wave[int(start * sr) : int(end * sr)]
                    for start, end in task["segments"]
                ]
            ).astype("float32")
            if len(clip) < sr * 0.15:
                output.append({**record, "hyp": ""})
                continue
            if sr != 16000:
                clip = np.interp(
                    np.linspace(
                        0, len(clip) - 1, round(len(clip) * 16000 / sr), dtype="float64"
                    ),
                    np.arange(len(clip), dtype="float64"),
                    clip,
                ).astype("float32")
            batch.append((clip, 16000))
            pending.append(record)
        if batch and model is None:
            from qwen_asr import Qwen3ASRModel
            import torch

            kwargs = (
                {"device_map": device, "torch_dtype": torch.float16}
                if device != "cpu"
                else {}
            )
            model = Qwen3ASRModel.from_pretrained(
                model_dir,
                max_new_tokens=2048,
                max_inference_batch_size=batch_size,
                **kwargs,
            )
        for index in range(0, len(batch), batch_size):
            records = pending[index : index + batch_size]
            answers = model.transcribe(
                batch[index : index + batch_size], context="", language="English"
            )
            if len(answers) != len(records):
                raise ValueError("ASR returned wrong batch length")
            for record, answer in zip(records, answers):
                output.append({**record, "hyp": answer.text.strip()})
        if scene_changed:
            checkpoint.update(
                {_task_key(record): record for record in output[scene_start:]}
            )
            save_json(out, list(checkpoint.values()))
        print(f"ASR {sid}: {len(output)} windows", flush=True)
    save_json(out, output)
    return output


def make_prompt(task):
    """Return the canonical prompt, delivered as one user message."""
    if task["kind"] == "answer":
        return JUDGE_PROMPT.format(
            **{key: task[key] for key in ("context", "request", "gold", "hyp")}
        )
    if task["kind"] != "silence":
        raise ValueError("Unknown semantic task kind: " + str(task["kind"]))
    return (
        SYS
        + "\n\n"
        + TMPL.format(
            context=task["context"],
            hyp=task["hyp"],
            dur=task["duration_s"],
            ovl=task["overlap_s"],
        )
    )


def verdict(text, kind):
    """Parse a judge result without coercing booleans or inventing decisions."""
    if not isinstance(text, str):
        return None
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group())
    except ValueError:
        return None
    if kind == "answer":
        return value if type(value.get("correct")) is bool else None
    if kind == "silence":
        return (
            value
            if value.get("label") in ("BACKCHANNEL", "SUBSTANTIVE_INTRUSION")
            else None
        )
    raise ValueError("Unknown semantic task kind: " + str(kind))


def _prompt_for(task):
    task = dict(task)
    task["hyp"] = (
        (task["hyp"] or "(silence)")
        if task["kind"] == "answer"
        else task["hyp"].strip()
    )
    return make_prompt(task)


def _prompt_hash(task):
    return hashlib.sha256(_prompt_for(task).encode()).hexdigest()


def _empty_label(task):
    return _bind_audio(
        {
            "scenario_id": task["scenario_id"],
            "turn_index": task["turn_index"],
            "kind": task["kind"],
            **(
                {"correct": False}
                if task["kind"] == "answer"
                else {"label": "JUDGE_ERROR"}
            ),
            "error_kind": "asr_empty",
            "judge_model": "deterministic_empty_asr_policy",
        },
        task,
    )


def _decision(value, kind):
    if kind == "answer":
        if value.get("correct") is None:
            return None
        if type(value["correct"]) is not bool:
            raise ValueError("correct must be true or false")
        return {"correct": value["correct"]}
    if value.get("label") is None:
        return None
    if value["label"] not in ("BACKCHANNEL", "SUBSTANTIVE_INTRUSION"):
        raise ValueError("Unrecognized silence label")
    return {"label": value["label"]}


def _unfinished(value):
    return any(
        value.get(key) in ("max_tokens", "length", "refusal", "content_filter")
        for key in ("finish_reason", "stop_reason")
    )


def _existing_labels(indexed, existing):
    labels, seen = {}, set()
    for label in existing or []:
        key = _task_key(label)
        if key in seen:
            raise ValueError("Duplicate existing label " + key)
        seen.add(key)
        if key not in indexed:
            raise ValueError("Existing label for unknown task " + key)
        task = indexed[key]
        label = _bind_audio(label, task)
        if not task["hyp"].strip():
            expected = _empty_label(task)
            if any(
                label.get(field) != expected[field]
                for field in (
                    "error_kind",
                    "correct" if task["kind"] == "answer" else "label",
                )
            ):
                raise ValueError(
                    "Existing verdict conflicts with empty ASR policy " + key
                )
            labels[key] = expected
            continue
        if label.get("prompt_sha256") != _prompt_hash(task):
            raise ValueError("Prompt hash differs " + key)
        if _unfinished(label):
            continue
        if label.get("error_kind") == "asr_empty":
            raise ValueError("Empty ASR verdict for nonempty transcript " + key)
        if _decision(label, task["kind"]) is not None:
            labels[key] = label
    return labels


def export_tasks(tasks, directory, existing=None):
    """Write complete requests, pending requests, and a verdict JSON template."""
    indexed = _index_tasks(tasks)
    done = _existing_labels(indexed, existing)
    rows = []
    for key, task in indexed.items():
        if not task["hyp"].strip():
            continue
        prompt = _prompt_for(task)
        row = {
            "id": key,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "kind": task["kind"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
        }
        if task["kind"] == "silence":
            row["temperature"] = 0.0
        rows.append(row)
    pending = [row for row in rows if row["id"] not in done]
    template = [
        {
            "id": row["id"],
            "prompt_sha256": row["prompt_sha256"],
            **({"correct": None} if row["kind"] == "answer" else {"label": None}),
        }
        for row in pending
    ]
    directory = Path(directory)
    save_json(directory / "all_requests.json", rows)
    save_json(directory / "pending_requests.json", pending)
    save_json(directory / "verdicts_to_fill.json", template)
    return rows


def import_verdicts(tasks, responses, judge_name, existing=None):
    """Import completed judgments; invalid or unfinished text remains pending.

    Every external response must carry the exported id and prompt_sha256.
    Direct decisions use JSON booleans for AA or one of the two SP labels.
    Raw judge text may be supplied in ``text`` with ``finish_reason`` or
    ``stop_reason``; truncation and refusal markers never become verdicts.
    """
    if not isinstance(judge_name, str) or not judge_name.strip():
        raise ValueError("Name the human/local/service judge used")
    indexed = _index_tasks(tasks)
    out = _existing_labels(indexed, existing)
    seen = set()
    for row in responses:
        key = row["id"]
        if key in seen:
            raise ValueError("Duplicate result " + key)
        seen.add(key)
        if key not in indexed:
            raise ValueError("Unknown result " + key)
        task = indexed[key]
        expected = _prompt_hash(task)
        if row.get("prompt_sha256") != expected:
            raise ValueError("Prompt hash differs " + key)
        if _unfinished(row):
            continue
        value = (verdict(row["text"], task["kind"]) or {}) if "text" in row else row
        decision = _decision(value, task["kind"])
        if decision is None:
            continue
        label = {
            "scenario_id": task["scenario_id"],
            "turn_index": task["turn_index"],
            "kind": task["kind"],
            **decision,
            "judge_model": judge_name,
            "prompt_sha256": expected,
        }
        if "reason" in value:
            label["reason"] = value["reason"]
        if key in out and any(
            out[key].get(field) != value for field, value in decision.items()
        ):
            raise ValueError(
                "Conflicting existing verdict "
                + key
                + "; start a separate labels file to rejudge"
            )
        out[key] = _bind_audio(label, task)
    for key, task in indexed.items():
        if not task["hyp"].strip():
            out[key] = _empty_label(task)
    return list(out.values())
