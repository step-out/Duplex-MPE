"""Causal Silero speech detection with the benchmark's original timing rules."""

from __future__ import annotations
from dataclasses import dataclass

_SILERO_MODEL = None


class _VoicingTracker:

    def _init_voicing(self) -> None:
        self.voiced_runs: list[tuple[float, float]] = []
        self._v_active = False
        self._v_run = 0
        self._v_start: int | None = None

    def _voicing_step(
        self, above_onset: bool, below_offset: bool, min_speech_frames: int
    ) -> None:
        if not self._v_active:
            if above_onset:
                self._v_run += 1
                if self._v_run >= min_speech_frames:
                    self._v_active = True
                    self._v_start = self.frames_seen - self._v_run
            else:
                self._v_run = 0
        elif below_offset:
            if self._v_start is not None:
                a = self._t(self._v_start)
                b = self._t(self.frames_seen - 1)
                if b > a:
                    self.voiced_runs.append((round(a, 4), round(b, 4)))
            self._v_active = False
            self._v_run = 0
            self._v_start = None

    def _finalize_voicing(self) -> None:
        if self._v_active and self._v_start is not None:
            a, b = (self._t(self._v_start), self._t(self.frames_seen))
            if b > a:
                self.voiced_runs.append((round(a, 4), round(b, 4)))
        self._v_active = False
        self._v_start = None

    @property
    def voicing_now(self) -> bool:
        return self._v_active

    def endpoint_confirmed(self, hangover_ms: float) -> bool:
        if self._v_active:
            return False
        e = self.last_voiced_end_s()
        if e is None:
            return False
        return self.now_s - e >= hangover_ms / 1000.0 - 1e-09

    def current_voicing_onset_s(self) -> float | None:
        return self._t(self._v_start) if self._v_start is not None else None

    def last_voiced_end_s(self) -> float | None:
        if self._v_active:
            return self._t(self.frames_seen)
        return self.voiced_runs[-1][1] if self.voiced_runs else None

    def voicing_onset_at_or_after(
        self, t0: float, since_index: int = 0
    ) -> float | None:
        for a, _b in self.voiced_runs[since_index:]:
            if a >= t0:
                return a
        if self._v_active and self._v_start is not None:
            a = self._t(self._v_start)
            if a >= t0:
                return a
        return None


def _silero_version() -> str:
    try:
        import silero_vad

        return str(getattr(silero_vad, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def get_silero_model():
    global _SILERO_MODEL
    if _SILERO_MODEL is None:
        from silero_vad import load_silero_vad

        _SILERO_MODEL = load_silero_vad()
    return _SILERO_MODEL


@dataclass
class SileroAEDConfig:
    onset_prob: float = 0.5
    offset_prob: float = 0.35
    hangover_ms: float = 800.0
    t_endpoint_hangover_ms: float = 2000.0
    min_speech_ms: float = 120.0
    silero_sr: int = 16000
    silero_window: int = 512

    @property
    def frame_ms(self) -> float:
        return 1000.0 * self.silero_window / self.silero_sr

    @property
    def frames_hangover(self) -> int:
        return max(1, int(round(self.hangover_ms / self.frame_ms)))

    @property
    def frames_min_speech(self) -> int:
        return max(1, int(round(self.min_speech_ms / self.frame_ms)))

    def as_dict(self) -> dict:
        return {
            "detector": "silero_streaming_hysteresis_hangover",
            "silero_vad_version": _silero_version(),
            "silero_sr": self.silero_sr,
            "silero_window": self.silero_window,
            "frame_ms": self.frame_ms,
            "onset_prob": self.onset_prob,
            "offset_prob": self.offset_prob,
            "hangover_ms": self.hangover_ms,
            "hangover_ms_role": "episode merge distance (M3 occupancy)",
            "t_endpoint_hangover_ms": self.t_endpoint_hangover_ms,
            "t_endpoint_hangover_role": "silence required to declare the T response finished; applied to raw voicing runs, NOT to episode merging",
            "min_speech_ms": self.min_speech_ms,
            "frames_hangover": self.frames_hangover,
            "frames_min_speech": self.frames_min_speech,
        }


class SileroAEDState(_VoicingTracker):

    def __init__(
        self,
        cfg: "SileroAEDConfig | None" = None,
        sr: int = 24000,
        model=None,
        rs_ctx_blocks: int = 3,
    ):
        import numpy as np
        import torch

        self._np, self._torch = (np, torch)
        self.cfg = cfg or SileroAEDConfig()
        self.sr = int(sr)
        self.model = model if model is not None else get_silero_model()
        self.model.reset_states()
        num = self.sr * self.cfg.silero_window
        if num % self.cfg.silero_sr != 0:
            raise ValueError(
                f"sr={self.sr} does not map to an integer number of native samples per {self.cfg.silero_window}-sample 16 kHz frame ({num}/{self.cfg.silero_sr}); supported examples: 16000, 24000, 32000, 48000"
            )
        self.frame_native = num // self.cfg.silero_sr
        from math import gcd

        g = gcd(self.cfg.silero_sr, self.sr)
        self._rs_up = self.cfg.silero_sr // g
        self._rs_down = self.sr // g
        self.rs_ctx_blocks = int(rs_ctx_blocks)
        self.in_speech = False
        self.speech_run = 0
        self.silence_run = 0
        self.frames_seen = 0
        self.episodes: list[tuple[float, float]] = []
        self._open_start_frame: int | None = None
        self._pending_end_frame: int | None = None
        self._carry = np.zeros(0, dtype="float32")
        self._rs_hist = np.zeros(0, dtype="float32")
        self._init_voicing()
        self.probs: list[float] = []

    def _t(self, frame_idx: int) -> float:
        return frame_idx * self.cfg.frame_ms / 1000.0

    @property
    def now_s(self) -> float:
        return self._t(self.frames_seen)

    @property
    def speaking(self) -> bool:
        return self.in_speech

    def open_episode_start_s(self) -> float | None:
        if self._open_start_frame is None:
            return None
        return self._t(self._open_start_frame)

    def _to_16k(self, block):
        np = self._np
        if self._rs_up == 1 and self._rs_down == 1:
            return block
        from scipy.signal import resample_poly

        hist = np.concatenate([self._rs_hist, block])
        out = resample_poly(hist.astype("float64"), self._rs_up, self._rs_down)
        keep = self.rs_ctx_blocks * self.frame_native
        self._rs_hist = hist[-keep:] if len(hist) > keep else hist
        w = self.cfg.silero_window
        if len(out) >= w:
            return out[-w:].astype("float32")
        return np.pad(out, (w - len(out), 0)).astype("float32")

    def push(self, samples) -> list[tuple[float, float]]:
        np, torch = (self._np, self._torch)
        x = np.asarray(samples, dtype="float32").reshape(-1)
        if len(self._carry):
            x = np.concatenate([self._carry, x])
            self._carry = np.zeros(0, dtype="float32")
        fl = self.frame_native
        n_full = len(x) // fl
        if n_full == 0:
            self._carry = x
            return []
        rest = x[n_full * fl :]
        if len(rest):
            self._carry = rest
        closed: list[tuple[float, float]] = []
        cfg = self.cfg
        for i in range(n_full):
            block = x[i * fl : (i + 1) * fl]
            frame16 = self._to_16k(block)
            with torch.no_grad():
                p = float(
                    self.model(
                        torch.from_numpy(np.ascontiguousarray(frame16))[None, :],
                        cfg.silero_sr,
                    )
                )
            self.probs.append(p)
            self.frames_seen += 1
            self._voicing_step(
                p >= cfg.onset_prob, p < cfg.offset_prob, cfg.frames_min_speech
            )
            if not self.in_speech:
                if p >= cfg.onset_prob:
                    self.speech_run += 1
                    if self.speech_run >= cfg.frames_min_speech:
                        self.in_speech = True
                        self.silence_run = 0
                        self._open_start_frame = self.frames_seen - self.speech_run
                else:
                    self.speech_run = 0
            elif p < cfg.offset_prob:
                self.silence_run += 1
                if self._pending_end_frame is None:
                    self._pending_end_frame = self.frames_seen - 1
                if self.silence_run >= cfg.frames_hangover:
                    start = self._t(self._open_start_frame or 0)
                    end = self._t(self._pending_end_frame or self.frames_seen)
                    if end > start:
                        seg = (round(start, 4), round(end, 4))
                        self.episodes.append(seg)
                        closed.append(seg)
                    self.in_speech = False
                    self.speech_run = 0
                    self.silence_run = 0
                    self._open_start_frame = None
                    self._pending_end_frame = None
            else:
                self.silence_run = 0
                self._pending_end_frame = None
        return closed

    def finalize(self) -> list[tuple[float, float]]:
        self._finalize_voicing()
        if self.in_speech and self._open_start_frame is not None:
            start = self._t(self._open_start_frame)
            end = self._t(self.frames_seen)
            if end > start:
                seg = (round(start, 4), round(end, 4))
                self.episodes.append(seg)
                self.in_speech = False
                self._open_start_frame = None
                return [seg]
        return []

    def stats(self) -> dict:
        np = self._np
        pr = np.asarray(self.probs, dtype="float64") if self.probs else None
        return {
            "detector": "silero_streaming",
            "frames": self.frames_seen,
            "episodes": len(self.episodes),
            "mean_prob": round(float(pr.mean()), 4) if pr is not None else None,
            "frac_frames_over_onset": (
                round(float((pr >= self.cfg.onset_prob).mean()), 4)
                if pr is not None
                else None
            ),
        }


def make_aed(config, sample_rate):
    if not isinstance(config, SileroAEDConfig):
        raise ValueError("The reference protocol requires the Silero detector")
    return SileroAEDState(config, sample_rate)
