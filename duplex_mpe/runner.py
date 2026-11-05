"""Run a user-supplied model on the reference audio-clock protocol."""

from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import random

import numpy as np
import soundfile as sf

from .dataset import load, load_json, resolve_audio, save_json
from .protocol import code_fingerprint, run_scenario
from .vad import SileroAEDConfig


@lru_cache(maxsize=1)
def _reference_seeds():
    return load_json(Path(__file__).with_name("seeds.json"))


def scenario_seed(scenario_id, seed_map=None):
    mapping = _reference_seeds() if seed_map is None else seed_map
    if scenario_id not in mapping:
        raise ValueError(
            "No reference seed for "
            + scenario_id
            + "; provide a seed map for custom data"
        )
    seed = mapping[scenario_id]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Scenario seed must be an integer in [0, 2**32)")
    return seed


class ClockGuard:
    """Require continuous output audio, including silence, on the input clock."""

    def __init__(self, model):
        self.model = model
        self.sr = int(model.sr)
        self.out_sr = int(getattr(model, "out_sr", self.sr))
        if self.sr <= 0 or self.out_sr <= 0:
            raise ValueError("Sample rates must be positive")
        self.nin = self.nout = 0

    def feed(self, chunk):
        output = self.model.feed(chunk)
        output = (
            np.array(output, dtype="float32", copy=True)
            if output is not None
            else np.empty(0, dtype="float32")
        )
        if output.ndim != 1 or not np.isfinite(output).all():
            raise ValueError("Output must be finite mono float32 audio")
        self.nin += len(chunk)
        self.nout += len(output)
        if abs(self.nout - self.nin * self.out_sr / self.sr) > 1.01:
            raise ValueError(
                "Model lost audio time: return silence when no output is available and queue excess audio"
            )
        return output


def _check_config(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if not key.endswith("_env") and any(
                token in key.lower()
                for token in (
                    "api_key",
                    "auth_token",
                    "access_token",
                    "secret",
                    "password",
                )
            ):
                raise ValueError(
                    "Read credentials from environment variables, not the saved model configuration"
                )
            _check_config(child)
    elif isinstance(value, list):
        for child in value:
            _check_config(child)


def run(root, out, model_spec, config=None, condition="all", limit=0, seed_map=None):
    if limit < 0:
        raise ValueError("limit cannot be negative")
    root, out = Path(root).resolve(), Path(out).resolve()
    if out == root or out.is_relative_to(root):
        raise ValueError("Write results outside the input dataset directory")
    config = {} if config is None else config
    if not isinstance(config, dict):
        raise ValueError("Model configuration must be a JSON object")
    _check_config(config)
    seeds = _reference_seeds() if seed_map is None else load_json(seed_map)
    selected = load(root, condition)
    if limit:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No scenarios selected")
    for scene in selected:
        scenario_seed(scene["scenario_id"], seeds)
    module, separator, factory_name = model_spec.partition(":")
    if not separator or not module or not factory_name:
        raise ValueError("Model entry point must be module:factory")
    fingerprint = code_fingerprint()
    run_config = {
        "model": model_spec,
        "config": config,
        "condition": condition,
        "code_fingerprint": fingerprint,
        "evaluation_clock": "audio_samples_excluding_compute_time",
        "seed_map_sha256": hashlib.sha256(
            json.dumps(seeds, sort_keys=True).encode()
        ).hexdigest(),
        "dataset_sha256": hashlib.sha256(
            (root / "dataset.json").read_bytes()
        ).hexdigest(),
        "manifest_sha256": {
            side: hashlib.sha256(
                (root / "manifests" / f"{side}.json").read_bytes()
            ).hexdigest()
            for side in ("explicit", "implicit")
        },
    }
    config_path, report_path = out / "run.json", out / "report.json"
    if config_path.exists() and load_json(config_path) != run_config:
        raise ValueError("Run configuration changed; use a new output directory")
    if report_path.exists() and not config_path.exists():
        raise ValueError("Existing report has no run configuration")
    save_json(config_path, run_config)
    reports = load_json(report_path) if report_path.exists() else {}
    factory = None
    for scene in selected:
        sid = scene["scenario_id"]
        audio_path = out / "audio" / f"{sid}.wav"
        if sid in reports:
            if not audio_path.exists() or hashlib.sha256(
                audio_path.read_bytes()
            ).hexdigest() != reports[sid].get("assistant_audio_sha256"):
                raise ValueError(
                    "Completed scenario audio is missing or changed: " + sid
                )
            continue
        if factory is None:
            factory = getattr(importlib.import_module(module), factory_name)(config)
            if not callable(factory):
                raise ValueError(
                    "factory(config) must return a callable that creates a fresh model"
                )
        seed = scenario_seed(sid, seeds)
        random.seed(seed)
        np.random.seed(seed)
        import torch

        torch.manual_seed(seed)
        model = factory()
        try:
            clock = ClockGuard(model)
            preamble, sample_rate = sf.read(
                resolve_audio(root, "audio/aria_duty_preamble.wav"), dtype="float32"
            )
            if preamble.ndim != 1:
                raise ValueError("Duty preamble must be mono")
            if sample_rate != clock.sr:
                preamble = np.interp(
                    np.linspace(
                        0,
                        len(preamble) - 1,
                        round(len(preamble) * clock.sr / sample_rate),
                    ),
                    np.arange(len(preamble)),
                    preamble,
                ).astype("float32")
            turns = [
                {
                    **turn,
                    "turn_audio_path": str(resolve_audio(root, turn["audio_path"])),
                }
                for turn in scene["turns"]
            ]
            with torch.no_grad():
                result = run_scenario(
                    clock,
                    turns,
                    aed_cfg=SileroAEDConfig(),
                    preamble=preamble,
                    collect_audio=True,
                )
            finished_model = model
            model = None
            if hasattr(finished_model, "close"):
                finished_model.close()
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = audio_path.with_name(audio_path.name + ".tmp")
            sf.write(
                temporary,
                result.assistant_wav,
                clock.out_sr,
                format="WAV",
                subtype="FLOAT",
            )
            temporary.replace(audio_path)
            record = result.as_dict()
            record.update(
                scenario_id=sid,
                pair_id=scene["pair_id"],
                addressing=scene["addressing"],
                run_seed=seed,
                code_fingerprint=fingerprint,
                assistant_wav_sr=clock.out_sr,
                evaluation_clock="audio_samples_excluding_compute_time",
                assistant_audio_sha256=hashlib.sha256(
                    audio_path.read_bytes()
                ).hexdigest(),
            )
            reports[sid] = record
            save_json(report_path, reports)
        finally:
            if model is not None and hasattr(model, "close"):
                model.close()
        print(f"{sid}: {record['T']['outcome']} ({len(reports)} finished)", flush=True)
    return reports
