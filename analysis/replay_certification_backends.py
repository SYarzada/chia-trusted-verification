from __future__ import annotations

import argparse
import json
from pathlib import Path

from chia_trusted_verification.backends.teaal_evolve.replay import replay_run


def _completed_run_dirs(root: Path):
    for manifest in sorted(root.rglob("run_manifest.json")):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("status") == "completed":
            yield manifest.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir")
    group.add_argument("--runs", help="Replay every completed run below this root")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.run_dir:
        output = replay_run(Path(args.run_dir), force=args.force)
        print(output)
        return

    root = Path(args.runs)
    run_dirs = list(_completed_run_dirs(root))
    print(f"completed runs to replay: {len(run_dirs)}")
    failures = 0
    for index, run_dir in enumerate(run_dirs, start=1):
        print(f"[{index}/{len(run_dirs)}] {run_dir}")
        try:
            output = replay_run(run_dir, force=args.force)
            print(f"  {output}")
        except Exception as exc:
            failures += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit(f"certification replay failed for {failures} run(s)")


if __name__ == "__main__":
    main()
