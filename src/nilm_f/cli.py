"""One pipeline: verified CSVs -> train-only preprocessing -> final F -> metrics."""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from .data import DEFAULT_SCOPE, load_clients
from .evaluation import evaluate


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_config(args, dataset, appliance, phase):
    resource = Path(__file__).resolve().parent / "resources" / "final_f.json"
    config = json.loads(resource.read_text(encoding="utf-8"))
    config.pop("method", None)
    config.update(dataset=dataset, appliance=appliance, phase=phase,
                  device=args.device, num_threads=args.threads, smoke=args.smoke)
    if args.smoke:
        config.update(rounds=2, finetune_passes=1)
    return config


def prepare(args, dataset, appliance, phase, output_dir):
    caps = dict(max_clients=2, max_train_windows=12, max_test_windows=4) if args.smoke else {}
    data = load_clients(args.data_root, dataset, appliance, phase=phase, **caps)
    write_json(Path(output_dir) / "preprocessing.json", {
        "dataset": dataset, "appliance": appliance, "phase": phase,
        "smoke": args.smoke, "households": [item["manifest"] for item in data],
        "data_scope": [item["scope_manifest"] for item in data],
        "loss_statistics": {item["household"]: item["aux_meta"] for item in data},
    })
    return data


def train_task(args, dataset, appliance, phase, output_dir):
    from .training import train
    config = load_config(args, dataset, appliance, phase)
    # Each phase constructs new phase-specific scalers and new model/optimizer
    # state. Formal retraining never starts from the development checkpoint.
    data = prepare(args, dataset, appliance, phase, output_dir)
    has_previous = (Path(output_dir) / "training_config.json").exists()
    result = train(data, config, output_dir, resume=args.resume and has_previous)
    del data
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return Path(output_dir) / "latest.pt"


def run_all(args):
    scope = json.loads(DEFAULT_SCOPE.read_text(encoding="utf-8"))
    results = []
    phases = ["formal"] if args.skip_development else ["development", "formal"]
    root = Path(args.output_root)
    for task in scope["tasks"]:
        dataset, appliance = task["dataset"], task["canonical"]
        for phase in phases:
            directory = root / dataset / appliance / phase
            print("Running {}/{} {}{}".format(dataset, appliance, phase,
                                              " [SMOKE]" if args.smoke else ""), flush=True)
            checkpoint = train_task(args, dataset, appliance, phase, directory)
            result = evaluate(checkpoint, args.data_root, directory / "evaluation", args.device,
                              save_predictions=args.save_predictions)
            results.append(result)
            write_json(root / "summary.json", {
                "method": "F", "seed": 1, "smoke": args.smoke,
                "development_included": not args.skip_development, "results": results,
            })
            with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
                fields = ["dataset", "appliance", "phase", "split", "mae_w", "f1_percent", "n_households", "n_points", "smoke"]
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(results)
    return {"summary": str(root / "summary.json"), "completed_phases": len(results)}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train", "evaluate", "run-all"):
        p = sub.add_parser(name)
        p.add_argument("--data-root", type=Path, required=True, help="Contains REDD/ and REFIT/ experiment CSVs")
        p.add_argument("--threads", type=int, default=4)
        if name != "prepare":
            p.add_argument("--device", default="cuda", help="cuda (default), cuda:0, or cpu")
        if name in ("prepare", "train"):
            p.add_argument("--dataset", choices=["REDD", "REFIT"], required=True)
            p.add_argument("--appliance", required=True, help="Fridge, DishWasher, Microwave, Washer, WashingMachine, or Kettle")
            p.add_argument("--phase", choices=["development", "formal"], default="formal")
            p.add_argument("--output-dir", type=Path, required=True)
        if name == "evaluate":
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--output-dir", type=Path, required=True)
        if name == "run-all":
            p.add_argument("--output-root", type=Path, required=True)
            p.add_argument("--skip-development", action="store_true", help="Run the already-fixed formal protocol only; not the original total development+retraining budget")
        if name in ("train", "run-all"):
            p.add_argument("--resume", action="store_true", help="Resume matching latest.pt if present")
        if name != "evaluate":
            p.add_argument("--smoke", action="store_true", help="Two households and tiny window/round budgets; NOT paper results")
        if name in ("evaluate", "run-all"):
            p.add_argument("--save-predictions", action="store_true", help="Write per-household NPZ arrays in watts")
    return parser


def main():
    args = build_parser().parse_args()
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "prepare":
        data = prepare(args, args.dataset, args.appliance, args.phase, args.output_dir)
        result = {"households": len(data), "manifest": str(args.output_dir / "preprocessing.json"), "smoke": args.smoke}
    elif args.command == "train":
        result = {"checkpoint": str(train_task(args, args.dataset, args.appliance, args.phase, args.output_dir))}
    elif args.command == "evaluate":
        result = evaluate(args.checkpoint, args.data_root, args.output_dir, args.device, args.save_predictions)
    else:
        result = run_all(args)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
