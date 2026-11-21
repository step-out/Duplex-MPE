# Duplex-MPE

Evaluation tools for **Duplex-MPE: Benchmarking Multi-Party Interaction
in Full-Duplex Dialogue**.

[Paper](https://arxiv.org/abs/2609.31948) ·
[Project page](https://step-out.github.io/Duplex-MPE-Page/) ·
[Dataset on Hugging Face](https://huggingface.co/datasets/ChengqianMa/Duplex-MPE) ·
[Dataset on ModelScope](https://www.modelscope.ai/datasets/ChengqianMa/Duplex-MPE) ·
[SGLang-Omni benchmark](https://sgl-project.github.io/sglang-omni/developer_reference/full_duplex_guide.html#multi-party-benchmark-duplex-mpe)

This package contains the shared scheduler, speech detector, six metrics, scored
ASR windows, and semantic-judge prompts. Connect your own model through the small
interface below, or evaluate live `/v1/realtime` servers directly in
[SGLang-Omni](#live-v1realtime-evaluation-in-sglang-omni)
([PR #4157](https://github.com/sgl-project/sglang-omni/pull/4157)). Model-specific
in-process weights, historical model outputs, dataset construction, and
paper-generation tools are not included.

## Install and check the data

Use Python 3.10 or newer, preferably in a virtual environment.

```bash
git clone https://github.com/step-out/Duplex-MPE.git
cd Duplex-MPE
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Download the dataset using its data-card instructions. It must contain
`dataset.json`, `manifests/`, `audio/`, and `audio_inventory.json`.

```bash
export DUPLEX_DATA=/path/to/Duplex-MPE-data
python -m duplex_mpe validate --data "$DUPLEX_DATA"
```

Add `--full-audio` to verify every audio checksum and header. The data files are
distributed separately; do not copy them into this code repository.

## Quick interface check

The included dummy returns silence and requires no evaluated-model weights.
Run these commands from the repository root:

```bash
python -m duplex_mpe run --data "$DUPLEX_DATA" \
  --model examples.dummy_model:factory --condition explicit --limit 1 \
  --out runs/sanity
python -m duplex_mpe score --data "$DUPLEX_DATA" \
  --report runs/sanity/report.json --out runs/sanity/scores.json
```

For this silent dummy, response rates are zero, silence preservation is one,
and conditional accuracy/yield are `null` because their denominators are zero.
This is an interface check, not a model-performance baseline.

## Connect and evaluate a model

Implement an importable `module:factory` following
[`examples/dummy_model.py`](examples/dummy_model.py):

- `factory(config)` returns a callable that creates fresh conversation state
  for each scenario. Shared weights can be loaded once inside the factory.
- The model exposes its input sample rate `sr` and output sample rate `out_sr`.
- `feed(audio)` receives a mono `float32` audio chunk and returns mono model audio.
  Cumulative output duration must match cumulative input duration. Return silence
  while no output is available, and queue excess output without dropping samples.
- `close()` is optional and releases the scenario state.

Only continuous audio reaches `feed`: the spoken duty preamble and the scheduled
human audio/silence. Do not supply transcripts, speaker labels, event labels,
gold answers or annotated boundaries to the evaluated model. Keep credentials
in environment variables, outside the saved model configuration.

```bash
python -m duplex_mpe run --data "$DUPLEX_DATA" \
  --model my_model:factory --config model.json --out runs/my-model
```

Make `my_model` importable from the current directory or your Python environment.
The output contains `run.json`, per-scenario records in `report.json`, and continuous
mono `FLOAT` WAV files in `audio/`. Repeating the same command resumes completed
scenarios after checking their audio hashes. Use a new output directory when
changing the model, weights, integration, code or configuration.

## Semantic scoring

Four metrics use timing alone. Conditional answer accuracy and voiced
silence-preservation windows also require ASR and semantic judgments.

```bash
pip install -e '.[asr]'
python -m duplex_mpe prepare --data "$DUPLEX_DATA" \
  --report runs/my-model/report.json --out runs/my-model/tasks.json
python -m duplex_mpe asr --tasks runs/my-model/tasks.json \
  --audio-dir runs/my-model/audio --out runs/my-model/transcribed.json
python -m duplex_mpe export-judge --tasks runs/my-model/transcribed.json \
  --out runs/my-model/judge
```

ASR uses `Qwen/Qwen3-ASR-1.7B`, English, empty ASR context, and at most 2,048 new
tokens. `--model` accepts a local model directory; `--device cpu` is also supported.
Detected speech segments are concatenated within each scored window before
transcription. Do not replace this with arbitrary whole-recording transcription.

Send the `messages` in `judge/pending_requests.json` to your chosen judge without
changing their contents. Both prompt types use a single **user** message;
[`prompts.py`](duplex_mpe/prompts.py) contains the exact templates. Preserve the
exported `max_tokens` and, when present, `temperature` settings.

Fill `judge/verdicts_to_fill.json` with these fields and save it as `judgments.json`:

| Task | Required decision |
|---|---|
| `answer` | `correct`: JSON `true` or `false` |
| `silence` | `label`: `BACKCHANNEL` or `SUBSTANTIVE_INTRUSION` |

Keep each exported `id` and `prompt_sha256`; `reason` is optional. Alternatively,
return raw judge output in `text`, carrying `finish_reason` or `stop_reason` when
available. Truncated/refused outputs do not count as decisions. Judge model and
version must be identified when importing:

```bash
python -m duplex_mpe import-judge --tasks runs/my-model/transcribed.json \
  --results judgments.json --judge-name YOUR_JUDGE_MODEL_AND_VERSION \
  --out runs/my-model/labels.json
python -m duplex_mpe score --data "$DUPLEX_DATA" \
  --report runs/my-model/report.json --labels runs/my-model/labels.json \
  --out runs/my-model/scores.json
```

Export/import accept `--existing-labels` for incremental judging. No speech in an
N1–N3 window passes directly. Detected speech with empty ASR is a deterministic
failure for semantic accuracy/SP; it is distinguished from a valid substantive-
intrusion judgment. Missing/invalid judgments remain pending rather than becoming
zero or a pass. Judge choice can change the semantic scores.
Imported labels retain the source-audio hash when available; scoring checks it
against the run report so decisions from different recordings cannot be mixed.

## Protocol and output

Each scenario starts from fresh model state: duty preamble, 0.5 s silence, then
human turns with ordinary 0.5 s gaps. After T, allow 5 s for a new onset; once
speech is present, wait for 2 s confirmed silence with a 60 s cap. After N4_Q,
stream 3 s silence and then N4_R regardless of model speech.

Timing uses the audio-sample clock, excluding model computation time. The
reference detector is Silero VAD 6.2.1 at 16 kHz with 512-sample frames, onset/
offset probabilities 0.5/0.35, 800 ms episode merging, 120 ms minimum speech,
and a separate 2,000 ms T-endpoint silence requirement. Output sample rates must
fit whole 32 ms frames, such as 16, 24, 32 or 48 kHz.

`seeds.json` maps the public scenario IDs to verified seeds from the shared
evaluation runs. Hashing the renamed public IDs would produce different seeds.
For custom scenarios, pass a JSON ID-to-seed map with `--seed-map`.

`report.json` is keyed by scenario ID. Each record contains `rows` with actual
human/scoring windows, `assistant_speech_segments`, `T`, detector settings, seed,
and the source-audio hash. The manifests alone do not define the model-dependent
timeline after T.

Scores are grouped by `addressing` (`explicit`/`implicit`), with four capabilities
and two coverage diagnostics. Each includes `numerator`, `denominator`, `pending`,
and `value` (a ratio). Zero denominators or pending judgments produce `null`.
Already-speaking responses count toward response presence and conditional
accuracy, but not fresh onset. Yield includes only eligible N4 events; silence
preservation excludes N4.

The release supports evaluating your own model and reproducing the metric
definitions. It does not provide one-command regeneration of the eight paper
systems. Run records identify this package's code fingerprint; historical run
fingerprints should remain unchanged when rescoring their saved reports.

## Live `/v1/realtime` evaluation in SGLang-Omni

Duplex-MPE is natively integrated into **[SGLang-Omni](https://github.com/sgl-project/sglang-omni)** ([PR #4157](https://github.com/sgl-project/sglang-omni/pull/4157)) for closed-loop evaluation of live `/v1/realtime` WebSocket servers across native full-duplex (`MiniCPM-o 4.5`, `PersonaPlex`, `Nemotron VoiceChat`) and unscored turn-based probe (`Qwen3-Omni`, `Streaming Cascade`) deployments:

- **Online Documentation & Runbook**: [Full-Duplex Guide — Multi-Party Benchmark: Duplex-MPE](https://sgl-project.github.io/sglang-omni/developer_reference/full_duplex_guide.html#multi-party-benchmark-duplex-mpe)
- **Benchmark CLI Guide**: [`benchmarks/README.md#duplex-mpe-multi-party-full-duplex-evaluation`](https://github.com/sgl-project/sglang-omni/blob/main/benchmarks/README.md#duplex-mpe-multi-party-full-duplex-evaluation)
- **Source Code**: [`benchmarks/duplex/mpe/`](https://github.com/sgl-project/sglang-omni/tree/main/benchmarks/duplex/mpe)

```bash
# Prepare the 40-stream smoke subset (or --dataset duplex-mpe for all 4,000 streams)
python -m benchmarks.dataset.prepare --dataset duplex-mpe-mini --local-dir benchmarks/cache/duplex_mpe_mini

# Evaluate a running /v1/realtime endpoint end-to-end
python -m benchmarks.eval.benchmark_duplex_mpe \
  --dataset-root benchmarks/cache/duplex_mpe_mini \
  --output-dir benchmarks/results/duplex_mpe_minicpmo \
  --mini \
  --model openbmb/MiniCPM-o-4_5 \
  --base-url http://127.0.0.1:8000 \
  --asr-base-url http://127.0.0.1:8001 \
  --judge-config benchmarks/configs/duplex_mpe_judge.example.json
```

## Citation

Code is distributed under the [MIT license](LICENSE). Cite the linked paper when
using the benchmark; the separately released dataset has its own license.

```bibtex
@article{ma2026duplexmpe,
  title   = {Duplex-MPE: Benchmarking Multi-Party Interaction in Full-Duplex Dialogue},
  author  = {Ma, Chengqian and Feng, Wenhao and Jin, Weixuan and Dai, Gaole and Xie, Tianyu and Ma, Yuexiao and Kang, Zhaolu and Zhao, Xiangyu and Zheng, Xiawu and Chao, Fei},
  journal = {arXiv preprint arXiv:2609.31948},
  year    = {2026}
}
```

When using the live `/v1/realtime` integration in [SGLang-Omni](https://github.com/sgl-project/sglang-omni), please also cite the SGLang-Omni runtime posts:

```bibtex
@misc{sglangomniinteractive2026,
  title        = {Unified Runtime for Full-Duplex Interactive Intelligence with {SGLang Omni}},
  author       = {{SGLang-Omni Team}},
  howpublished = {\url{https://www.sglang.io/blog/omni-interactive-runtime}},
  year         = {2026},
  month        = {oct}
}

@misc{sglangomni2026,
  title        = {{SGLang Omni}: Serving Multi-Stage Omni Models with Heterogeneous Stage Scheduling},
  author       = {{SGLang-Omni Team}},
  howpublished = {\url{https://www.sglang.io/blog/spg-topology}},
  year         = {2026},
  month        = {oct}
}
```
