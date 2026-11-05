"""Six benchmark metrics with explicit denominators and pending judgments."""

from __future__ import annotations
from .protocol import T_HIT, T_MISS, T_OVERLONG, T_HIT_CONTINUATION

NEG = ("N1", "N2", "N3")
T_RESPONDED = (T_HIT, T_HIT_CONTINUATION, T_OVERLONG)
EPS = 1e-6


def first_onset(segs, a, b, min_voiced):
    best = None
    for s0, e0 in segs:
        if e0 <= a or s0 >= b:
            continue
        if min(e0, b) - max(s0, a) < min_voiced and best is None:
            continue
        c = max(s0, a)
        best = c if best is None else min(best, c)
    return best


def crosses(segs, t):
    return any((s0 <= t + EPS < e0 for s0, e0 in segs))


def voiced_between(segs, a, b):
    if b <= a + EPS:
        return False
    return any((e0 > a + EPS and s0 < b - EPS for s0, e0 in segs))


def get_segments(v: dict):
    for k in (
        "assistant_speech_segments",
        "speech_segments",
        "output_speech_segments",
        "assistant_episodes",
    ):
        segs = v.get(k)
        if segs:
            return [(float(a), float(b)) for a, b in segs]
    return []


def overlap_dur(segs, a: float, b: float) -> float:
    return sum((max(0.0, min(e, b) - max(s, a)) for s, e in segs))


def segs_in(segs, a: float, b: float):
    out = []
    for s, e in segs:
        s2, e2 = (max(s, a), min(e, b))
        if e2 > s2:
            out.append((round(s2, 4), round(e2, 4)))
    return out


def has_onset(segs, a: float, b: float) -> bool:
    return any((a <= s < b for s, _e in segs))


def timing_class(t: dict) -> str | None:
    oc = t.get("outcome")
    if not oc:
        return None
    if oc in (T_HIT, T_MISS, T_HIT_CONTINUATION):
        return oc
    if oc == T_OVERLONG:
        if t.get("speaking_at_t_end"):
            return T_HIT_CONTINUATION
        if t.get("onset_s") is not None:
            return T_HIT
        return "T_OVERLONG_UNATTRIBUTABLE"
    return oc


def collect_non_t_episodes(scen: dict, scenarios_by_id: dict | None):
    eps = []
    for sid, v in scen.items():
        segs = get_segments(v)
        turns = None
        if scenarios_by_id and sid in scenarios_by_id:
            turns = scenarios_by_id[sid].get("turns")
        for r in v.get("rows", []):
            if r.get("event_type") not in NEG:
                continue
            a, b = (r.get("win_start"), r.get("win_end"))
            if a is None or b is None:
                continue
            inside = segs_in(segs, a, b)
            if not inside:
                continue
            ctx = ""
            if turns:
                i = r.get("turn_index")
                if isinstance(i, int):
                    lo = max(0, i - 2)
                    ctx = "\n".join(
                        (
                            f"  {t.get('speaker', '?')}: {t.get('text', '')}"
                            for t in turns[lo : i + 1]
                        )
                    )
            eps.append(
                {
                    "scenario_id": sid,
                    "turn_index": r.get("turn_index"),
                    "event_type": r.get("event_type"),
                    "win_start": a,
                    "win_end": b,
                    "segments": inside,
                    "duration_s": round(overlap_dur(segs, a, b), 4),
                    "overlap_s": round(
                        overlap_dur(segs, r.get("turn_start", a), r.get("turn_end", b)),
                        4,
                    ),
                    "onset_in_window": has_onset(segs, a, b),
                    "context": ctx,
                }
            )
    return eps


def collect_m2_cases(scen: dict) -> list:
    cases = []
    for sid, v in scen.items():
        t = v.get("T") or {}
        if t.get("outcome") not in T_RESPONDED:
            continue
        segs = get_segments(v)
        row = next((r for r in v.get("rows", []) if r.get("event_type") == "T"), None)
        if row is None:
            continue
        a = row.get("win_start")
        b = row.get("win_end")
        if a is None or b is None:
            continue
        inside = segs_in(segs, a, b)
        if not inside:
            continue
        cases.append(
            {
                "scenario_id": sid,
                "turn_index": row.get("turn_index"),
                "win_start": a,
                "win_end": b,
                "segments": inside,
                "t_outcome": t.get("outcome"),
            }
        )
    return cases


def n4_counts(report):
    segs = get_segments(report)
    rows = report["rows"]
    byidx = {r["turn_index"]: r for r in rows}
    out = dict(windows=0, window_hits=0, eligible=0, yielded=0)
    for q in rows:
        if q["event_type"] != "N4_Q":
            continue
        r = byidx[q["turn_index"] + 1]
        if r["event_type"] != "N4_R":
            raise ValueError("N4 question must be followed by resolution")
        out["windows"] += 1
        onset = first_onset(segs, q["win_start"], q["win_end"], 0.12)
        if onset is None:
            continue
        out["window_hits"] += 1
        # Preserve the original 3-second pause boundary and earliest-onset rule.
        if abs(onset - q["win_start"]) < 1e-6 or onset < q["win_end"] - 3 - 1e-6:
            continue
        start = r["turn_start"]
        if not crosses(segs, start):
            continue
        out["eligible"] += 1
        if not crosses(segs, start + 2) and not voiced_between(
            segs, start + 2, r["win_end"]
        ):
            out["yielded"] += 1
    return out


def ratio(num, den, pending=0):
    return {
        "numerator": num,
        "denominator": den,
        "pending": pending,
        "value": num / den if den and not pending else None,
    }


def score(reports, scenes, labels):
    indexed = {}
    for x in labels:
        key = (x["scenario_id"], x["turn_index"], x["kind"])
        if key in indexed:
            raise ValueError("Duplicate semantic verdict: " + str(key))
        # New run records bind decisions to the waveform that was transcribed.
        # Historical reports without a recorded hash remain scoreable.
        expected_audio = reports.get(x["scenario_id"], {}).get("assistant_audio_sha256")
        decided = (
            type(x.get("correct")) is bool
            or x.get("label") in ("BACKCHANNEL", "SUBSTANTIVE_INTRUSION")
            or x.get("error_kind") == "asr_empty"
        )
        if expected_audio and (decided or x.get("assistant_audio_sha256")):
            if x.get("assistant_audio_sha256") != expected_audio:
                raise ValueError("Label source audio differs from report: " + str(key))
        indexed[key] = x
    groups = {}
    for sid, r in reports.items():
        if sid not in scenes:
            raise ValueError("Unknown scenario in report: " + sid)
        s = scenes[sid]
        c = s["addressing"]
        counts = groups.setdefault(
            c,
            dict(
                n=0,
                fresh=0,
                responded=0,
                correct=0,
                answer_pending=0,
                silence_n=0,
                silence_ok=0,
                silence_pending=0,
                windows=0,
                window_hits=0,
                eligible=0,
                yielded=0,
            ),
        )
        counts["n"] += 1
        cls = timing_class(r["T"])
        if cls not in ("T_HIT_5S", "T_HIT_CONTINUATION", "T_MISS_5S"):
            raise ValueError("Missing/invalid T outcome")
        counts["fresh"] += int(cls == "T_HIT_5S")
        responded = cls != "T_MISS_5S"
        counts["responded"] += responded
        if responded:
            cases = collect_m2_cases({sid: r})
            if len(cases) != 1:
                raise ValueError("Response marked present but no scored speech: " + sid)
            x = indexed.get((sid, cases[0]["turn_index"], "answer"), {})
            if type(x.get("correct")) is not bool:
                counts["answer_pending"] += 1
            else:
                counts["correct"] += x["correct"]
        windows = [x for x in r["rows"] if x["event_type"] in ("N1", "N2", "N3")]
        counts["silence_n"] += len(windows)
        episodes = collect_non_t_episodes({sid: r}, {sid: s})
        counts["silence_ok"] += len(windows) - len(episodes)
        for ep in episodes:
            x = indexed.get((sid, ep["turn_index"], "silence"), {})
            if x.get("label") == "BACKCHANNEL":
                counts["silence_ok"] += 1
            elif (
                x.get("label") == "SUBSTANTIVE_INTRUSION"
                or x.get("error_kind") == "asr_empty"
            ):
                pass
            else:
                counts["silence_pending"] += 1
        for k, v in n4_counts(r).items():
            counts[k] += v
    return {
        c: {
            "scenarios": n["n"],
            "fresh_onset_response_rate": ratio(n["fresh"], n["n"]),
            "conditional_answer_accuracy": ratio(
                n["correct"], n["responded"], n["answer_pending"]
            ),
            "silence_preservation": ratio(
                n["silence_ok"], n["silence_n"], n["silence_pending"]
            ),
            "answering_window_yield": ratio(n["yielded"], n["eligible"]),
            "response_presence": ratio(n["responded"], n["n"]),
            "window_response": ratio(n["window_hits"], n["windows"]),
        }
        for c, n in groups.items()
    }


def validate_reports(reports, scenes):
    """Reject partial conversations or incompatible timing configurations."""
    import math

    if not reports:
        raise ValueError("No completed conversations in report")
    for sid, r in reports.items():
        if sid not in scenes:
            raise ValueError("Unknown scenario: " + sid)
        if r.get("protocol_id") != "mpeval.unified.v3":
            raise ValueError("Incompatible evaluation protocol")
        expected = [(i, t["event_type"]) for i, t in enumerate(scenes[sid]["turns"])]
        actual = [(x["turn_index"], x["event_type"]) for x in r.get("rows", [])]
        if actual != expected:
            raise ValueError("Missing, reordered or mislabeled turns: " + sid)
        duration = r.get("total_duration_s", 0)
        if (
            not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError("Missing total duration")
        for row in r["rows"]:
            times = [row[k] for k in ("turn_start", "turn_end", "win_start", "win_end")]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in times):
                raise ValueError("Invalid time values")
            if not 0 <= row["turn_start"] <= row["turn_end"] <= duration + 0.04:
                raise ValueError("Human turn outside audio timeline")
            if not 0 <= row["win_start"] <= row["win_end"] <= duration + 0.04:
                raise ValueError("Scoring window outside audio timeline")
        previous = 0.0
        for a, b in get_segments(r):
            if not 0 <= a < b <= duration + 0.04 or a < previous - 1e-6:
                raise ValueError("Invalid speech segments")
            previous = b
        cfg = r.get("aed_config", {})
        required = {
            "detector": "silero_streaming_hysteresis_hangover",
            "silero_vad_version": "6.2.1",
            "silero_sr": 16000,
            "onset_prob": 0.5,
            "offset_prob": 0.35,
            "hangover_ms": 800.0,
            "t_endpoint_hangover_ms": 2000.0,
            "min_speech_ms": 120.0,
            "silero_window": 512,
        }
        if any(cfg.get(k) != v for k, v in required.items()):
            raise ValueError("Different output detector settings: " + sid)
