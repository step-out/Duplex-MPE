"""Model-independent playback protocol and audio-clock timing windows."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
from .vad import SileroAEDConfig, make_aed

T_ONSET_DEADLINE_S = 5.0
T_MAX_RESPONSE_S = 60.0
N4_PAUSE_S = 3.0
NON_T_GAP_S = 0.5
PROTOCOL_ID = "mpeval.unified.v3"
T_HIT = "T_HIT_5S"
T_MISS = "T_MISS_5S"
T_OVERLONG = "T_OVERLONG_60S"
T_HIT_CONTINUATION = "T_HIT_CONTINUATION"


def code_fingerprint():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update((root / "seeds.json").read_bytes())
    return digest.hexdigest()[:12]


@dataclass
class TurnWindow:
    index: int
    event_type: str
    speaker: str
    start_s: float
    end_s: float
    win_start: float
    win_end: float


@dataclass
class ScheduleResult:
    protocol_id: str
    windows: list[TurnWindow] = field(default_factory=list)
    assistant_episodes: list[tuple[float, float]] = field(default_factory=list)
    t_outcome: str | None = None
    t_end_s: float | None = None
    t_onset_s: float | None = None
    t_onset_latency_s: float | None = None
    t_endpoint_s: float | None = None
    t_response_duration_s: float | None = None
    t_forced_return: bool = False
    speaking_at_t_end: bool = False
    total_duration_s: float = 0.0
    aed_config: dict = field(default_factory=dict)
    assistant_wav: object = None

    def as_dict(self) -> dict:
        return {
            "protocol_id": self.protocol_id,
            "aed_config": self.aed_config,
            "total_duration_s": round(self.total_duration_s, 4),
            "assistant_speech_segments": [[a, b] for a, b in self.assistant_episodes],
            "T": {
                "outcome": self.t_outcome,
                "t_end_s": self.t_end_s,
                "onset_s": self.t_onset_s,
                "onset_latency_s": self.t_onset_latency_s,
                "endpoint_s": self.t_endpoint_s,
                "response_duration_s": self.t_response_duration_s,
                "forced_floor_return": self.t_forced_return,
                "speaking_at_t_end": self.speaking_at_t_end,
            },
            "rows": [
                {
                    "turn_index": w.index,
                    "event_type": w.event_type,
                    "speaker": w.speaker,
                    "turn_start": round(w.start_s, 4),
                    "turn_end": round(w.end_s, 4),
                    "win_start": round(w.win_start, 4),
                    "win_end": round(w.win_end, 4),
                }
                for w in self.windows
            ],
        }


def _zeros(n: int):
    import numpy as np

    return np.zeros(max(0, n), dtype="float32")


def run_scenario(
    adapter,
    turns,
    aed_cfg: SileroAEDConfig | None = None,
    chunk_ms: float = 80.0,
    non_t_gap_s: float = NON_T_GAP_S,
    n4_pause_s: float = N4_PAUSE_S,
    onset_deadline_s: float = T_ONSET_DEADLINE_S,
    max_response_s: float = T_MAX_RESPONSE_S,
    t_endpoint_hangover_ms: float | None = None,
    preamble=None,
    progress=None,
    collect_audio: bool = False,
) -> ScheduleResult:
    import numpy as np, soundfile as sf

    sr = int(adapter.sr)
    cfg = aed_cfg or SileroAEDConfig()
    out_sr = int(getattr(adapter, "out_sr", sr) or sr)
    aed = make_aed(cfg, out_sr)
    if t_endpoint_hangover_ms is None:
        t_endpoint_hangover_ms = float(
            getattr(cfg, "t_endpoint_hangover_ms", cfg.hangover_ms)
        )
    res = ScheduleResult(protocol_id=PROTOCOL_ID, aed_config=cfg.as_dict())
    res.aed_config["code_fingerprint"] = code_fingerprint()
    res.aed_config["t_endpoint_hangover_ms"] = float(t_endpoint_hangover_ms)
    res.aed_config["input_sr"] = sr
    res.aed_config["assistant_sr"] = out_sr
    chunk = max(1, int(round(sr * chunk_ms / 1000.0)))

    def load(t) -> "np.ndarray":
        if t.get("wav") is not None:
            w = np.asarray(t["wav"], dtype="float32").reshape(-1)
            return w
        p = t.get("turn_audio_path")
        if not p or not Path(p).exists():
            raise FileNotFoundError(f"turn audio missing: {p}")
        w, s = sf.read(p, dtype="float32")
        if getattr(w, "ndim", 1) > 1:
            w = w.mean(axis=1)
        if s != sr:
            n = int(round(len(w) * sr / s))
            w = np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w).astype(
                "float32"
            )
        return w

    _cap: list = []

    def _sink(out) -> None:
        if out is not None and len(out):
            aed.push(out)
            if collect_audio:
                _cap.append(np.asarray(out, dtype="float32").reshape(-1))

    def feed_audio(x) -> None:
        for i in range(0, len(x), chunk):
            _sink(adapter.feed(x[i : i + chunk]))

    def feed_silence_until(stop_predicate, limit_s: float) -> float:
        total = int(round(limit_s * sr))
        fed = 0
        while fed < total:
            n = min(chunk, total - fed)
            _sink(adapter.feed(_zeros(n)))
            fed += n
            if stop_predicate():
                break
        return fed / sr

    if preamble is not None and len(preamble):
        feed_audio(np.asarray(preamble, dtype="float32").reshape(-1))
        feed_silence_until(lambda: False, non_t_gap_s)
    for idx, t in enumerate(turns):
        et = t.get("event_type")
        wav = load(t)
        start = aed.now_s
        feed_audio(wav)
        end = aed.now_s
        if et != "T":
            gap = n4_pause_s if et == "N4_Q" else non_t_gap_s
            feed_silence_until(lambda: False, gap)
            res.windows.append(
                TurnWindow(
                    index=idx,
                    event_type=et,
                    speaker=t.get("speaker", "?"),
                    start_s=start,
                    end_s=end,
                    win_start=start,
                    win_end=aed.now_s,
                )
            )
            if progress:
                progress(idx, et, None)
            continue
        res.t_end_s = round(end, 4)
        was_speaking_at_t_end = aed.voicing_now
        n_eps_at_t_end = len(aed.episodes)
        n_vruns_at_t_end = len(aed.voiced_runs)
        res.speaking_at_t_end = bool(was_speaking_at_t_end)
        res.aed_config.setdefault("t_end_state", {})
        res.aed_config["t_end_state"] = {
            "voicing_now": bool(was_speaking_at_t_end),
            "episode_open": bool(aed.speaking),
            "disagree_due_to_hangover": bool(
                aed.speaking and (not was_speaking_at_t_end)
            ),
        }

        def onset_seen() -> bool:
            return (
                aed.voicing_onset_at_or_after(end, since_index=n_vruns_at_t_end)
                is not None
            )

        if was_speaking_at_t_end:

            def finished() -> bool:
                return aed.endpoint_confirmed(t_endpoint_hangover_ms)

            fed = feed_silence_until(finished, max_response_s)
            capped = fed >= max_response_s - 1e-06
            if capped and (not aed.endpoint_confirmed(t_endpoint_hangover_ms)):
                res.t_outcome = T_OVERLONG
                res.t_forced_return = True
                res.t_response_duration_s = round(max_response_s, 4)
            else:
                res.t_outcome = T_HIT_CONTINUATION
                vend = aed.last_voiced_end_s()
                if vend is None or vend < end:
                    vend = aed.episodes[-1][1] if aed.episodes else aed.now_s
                res.t_endpoint_s = round(float(vend), 4)
                res.t_response_duration_s = round(float(vend) - end, 4)
            res.windows.append(
                TurnWindow(
                    index=idx,
                    event_type=et,
                    speaker=t.get("speaker", "?"),
                    start_s=start,
                    end_s=end,
                    win_start=start,
                    win_end=aed.now_s,
                )
            )
            if progress:
                progress(idx, et, res.t_outcome)
            continue
        feed_silence_until(onset_seen, onset_deadline_s)
        if not onset_seen():
            res.t_outcome = T_MISS
            res.windows.append(
                TurnWindow(
                    index=idx,
                    event_type=et,
                    speaker=t.get("speaker", "?"),
                    start_s=start,
                    end_s=end,
                    win_start=start,
                    win_end=aed.now_s,
                )
            )
            if progress:
                progress(idx, et, T_MISS)
            continue
        onset = aed.voicing_onset_at_or_after(end, since_index=n_vruns_at_t_end)
        if onset is None:
            onset = end
        onset = float(onset)
        res.t_onset_s = round(onset, 4)
        res.t_onset_latency_s = round(onset - end, 4)

        def endpoint_seen() -> bool:
            return aed.endpoint_confirmed(t_endpoint_hangover_ms)

        remaining = max(0.0, onset + max_response_s - aed.now_s)
        feed_silence_until(endpoint_seen, remaining)
        if endpoint_seen():
            endpoint = aed.last_voiced_end_s()
            if endpoint is None or endpoint < onset:
                endpoint = aed.episodes[-1][1] if aed.episodes else aed.now_s
            res.t_outcome = T_HIT
            res.t_endpoint_s = round(float(endpoint), 4)
            res.t_response_duration_s = round(float(endpoint) - onset, 4)
        else:
            res.t_outcome = T_OVERLONG
            res.t_forced_return = True
            res.t_response_duration_s = round(max_response_s, 4)
        res.windows.append(
            TurnWindow(
                index=idx,
                event_type=et,
                speaker=t.get("speaker", "?"),
                start_s=start,
                end_s=end,
                win_start=start,
                win_end=aed.now_s,
            )
        )
        if progress:
            progress(idx, et, res.t_outcome)
    aed.finalize()
    if collect_audio:
        res.assistant_wav = (
            np.concatenate(_cap) if _cap else np.zeros(0, dtype="float32")
        )
    res.assistant_episodes = list(aed.episodes)
    res.total_duration_s = aed.now_s
    return res
