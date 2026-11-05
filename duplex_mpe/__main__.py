"""Command-line entry point for the minimal evaluation workflow."""

import argparse
import json

from . import __version__
from .dataset import load, load_json, save_json, validate


def parser():
    command = argparse.ArgumentParser(prog="duplex-mpe")
    command.add_argument("--version", action="version", version=__version__)
    sub = command.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate", help="validate the downloaded dataset")
    check.add_argument("--data", required=True)
    check.add_argument(
        "--full-audio",
        action="store_true",
        help="also verify every audio checksum and header",
    )
    run = sub.add_parser("run", help="evaluate a user-supplied model")
    run.add_argument("--data", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--model", required=True, help="importable module:factory")
    run.add_argument("--config", help="JSON model settings, without credentials")
    run.add_argument(
        "--condition", choices=["all", "explicit", "implicit"], default="all"
    )
    run.add_argument(
        "--limit", type=int, default=0, help="limit scenarios; 0 means all"
    )
    run.add_argument(
        "--seed-map", help="optional public scenario ID to integer seed JSON"
    )
    for name in ["prepare", "score"]:
        item = sub.add_parser(
            name,
            help=(
                "prepare semantic tasks"
                if name == "prepare"
                else "compute all six metrics"
            ),
        )
        item.add_argument("--data", required=True)
        item.add_argument("--report", required=True)
        item.add_argument("--out", required=True)
        if name == "score":
            item.add_argument(
                "--labels",
                help="imported semantic labels; missing labels remain pending",
            )
    asr = sub.add_parser(
        "asr", help="transcribe scored speech segments using Qwen3-ASR"
    )
    asr.add_argument("--tasks", required=True)
    asr.add_argument("--audio-dir", required=True)
    asr.add_argument("--out", required=True)
    asr.add_argument(
        "--model", default="Qwen/Qwen3-ASR-1.7B", help="ASR model ID or local directory"
    )
    asr.add_argument("--device", default="cuda")
    asr.add_argument("--batch-size", type=int, default=16)
    export = sub.add_parser(
        "export-judge", help="export exact prompts for an external judge"
    )
    export.add_argument("--tasks", required=True, help="tasks after ASR")
    export.add_argument("--out", required=True)
    export.add_argument("--existing-labels")
    ingest = sub.add_parser(
        "import-judge", help="validate and import external judgments"
    )
    ingest.add_argument("--tasks", required=True, help="the same tasks used for export")
    ingest.add_argument("--results", required=True)
    ingest.add_argument(
        "--judge-name",
        required=True,
        help="judge model/version or human annotation identifier",
    )
    ingest.add_argument("--existing-labels")
    ingest.add_argument("--out", required=True)
    return command


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    try:
        if args.command == "validate":
            result = validate(args.data, args.full_audio)
        elif args.command == "run":
            from .runner import run

            reports = run(
                args.data,
                args.out,
                args.model,
                load_json(args.config) if args.config else None,
                args.condition,
                args.limit,
                args.seed_map,
            )
            result = {"completed_scenarios": len(reports), "output": args.out}
        elif args.command in {"prepare", "score"}:
            from .metrics import score, validate_reports

            reports = load_json(args.report)
            scenes = {scene["scenario_id"]: scene for scene in load(args.data)}
            validate_reports(reports, scenes)
            if any(report.get("smoke_test") for report in reports.values()):
                raise ValueError(
                    "Debug-detector reports cannot be used as reference benchmark results"
                )
            if args.command == "prepare":
                from .semantic import tasks_for

                tasks = tasks_for(reports, scenes)
                save_json(args.out, tasks)
                result = {"tasks": len(tasks), "output": args.out}
            else:
                result = score(
                    reports, scenes, load_json(args.labels) if args.labels else []
                )
                save_json(args.out, result)
        elif args.command == "asr":
            from .semantic import transcribe

            tasks = transcribe(
                load_json(args.tasks),
                args.audio_dir,
                args.out,
                args.model,
                args.device,
                args.batch_size,
            )
            result = {"transcribed_tasks": len(tasks), "output": args.out}
        elif args.command == "export-judge":
            from .semantic import export_tasks

            rows = export_tasks(
                load_json(args.tasks),
                args.out,
                load_json(args.existing_labels) if args.existing_labels else None,
            )
            result = {"requests": len(rows), "output": args.out}
        else:
            from .semantic import import_verdicts

            labels = import_verdicts(
                load_json(args.tasks),
                load_json(args.results),
                args.judge_name,
                load_json(args.existing_labels) if args.existing_labels else None,
            )
            save_json(args.out, labels)
            result = {"labels": len(labels), "output": args.out}
    except (ValueError, KeyError, FileNotFoundError, ImportError) as error:
        command.exit(2, f"duplex-mpe: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
