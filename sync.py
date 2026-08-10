#!/usr/bin/env python3
import argparse
import datetime as dt
import re
import secrets
import string
import subprocess
import sys
from pathlib import Path

RUN_NAME_RE = re.compile(
    r"^(?:offline-run|run)-(?P<date>\d{8})_(?P<time>\d{6})(?:-[A-Za-z0-9]+)?$"
)

RETRY_WITH_NEW_ID_HINTS = [
    "cannot be reused",
    "already exists",
    "conflict",
    "409",
    "run id",
    "deleted",
]


def parse_run_time(run_dir: Path):
    """
    Ưu tiên parse thời gian từ tên thư mục:
      offline-run-YYYYMMDD_HHMMSS-xxxx
      run-YYYYMMDD_HHMMSS-xxxx
    Nếu không parse được thì fallback sang mtime của thư mục.
    """
    m = RUN_NAME_RE.match(run_dir.name)
    if m:
        try:
            s = f"{m.group('date')}_{m.group('time')}"
            return dt.datetime.strptime(s, "%Y%m%d_%H%M%S")
        except ValueError:
            pass

    return dt.datetime.fromtimestamp(run_dir.stat().st_mtime)


def is_valid_run_dir(run_dir: Path):
    if not run_dir.is_dir():
        return False
    if not (run_dir.name.startswith("offline-run-") or run_dir.name.startswith("run-")):
        return False

    # Có file .wandb bên trong thì mới coi là run hợp lệ
    wandb_files = list(run_dir.glob("run-*.wandb"))
    return len(wandb_files) > 0


def gen_new_run_id(length=8):
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def run_command(cmd, dry_run=False):
    print(f"\n[CMD] {' '.join(cmd)}")
    if dry_run:
        return 0, ""

    result = subprocess.run(cmd, text=True, capture_output=True)
    output = (result.stdout or "") + (result.stderr or "")
    if output.strip():
        print(output.strip())
    return result.returncode, output


def should_retry_with_new_id(output: str):
    out = output.lower()
    return any(hint in out for hint in RETRY_WITH_NEW_ID_HINTS)


def sync_run_original_id(run_dir: Path, include_synced=False, entity=None, project=None, dry_run=False):
    cmd = ["wandb", "sync", str(run_dir)]

    if include_synced:
        cmd.insert(2, "--include-synced")
    if entity:
        cmd.extend(["-e", entity])
    if project:
        cmd.extend(["-p", project])

    return run_command(cmd, dry_run=dry_run)


def sync_run_new_id(run_dir: Path, entity=None, project=None, dry_run=False):
    new_id = gen_new_run_id()
    cmd = ["wandb", "sync", str(run_dir), "--id", new_id]

    if entity:
        cmd.extend(["-e", entity])
    if project:
        cmd.extend(["-p", project])

    code, output = run_command(cmd, dry_run=dry_run)
    return code, output, new_id


def main():
    parser = argparse.ArgumentParser(
        description="Khôi phục các W&B runs local lên trang W&B cho các run trong N ngày gần đây."
    )
    parser.add_argument(
        "--wandb-dir",
        default="wandb",
        help="Đường dẫn tới thư mục wandb (mặc định: ./wandb)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=21,
        help="Chỉ sync run trong vòng N ngày gần đây (mặc định: 21)",
    )
    parser.add_argument(
        "--include-synced",
        action="store_true",
        help="Bao gồm cả các run đã từng được đánh dấu synced",
    )
    parser.add_argument(
        "--entity",
        default="aiotlab",
        help="W&B entity/team",
    )
    parser.add_argument(
        "--project",
        default="SubSeasonalForecasting",
        help="W&B project",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ in ra lệnh sync, không chạy thật",
    )
    parser.add_argument(
        "--always-new-id",
        action="store_true",
        help="Luôn tạo ID mới khi sync, không thử ID cũ",
    )
    args = parser.parse_args()

    wandb_dir = Path(args.wandb_dir).resolve()
    if not wandb_dir.exists() or not wandb_dir.is_dir():
        print(f"[ERROR] Không tìm thấy thư mục: {wandb_dir}", file=sys.stderr)
        sys.exit(1)

    now = dt.datetime.now()
    cutoff = now - dt.timedelta(days=args.days)

    candidates = []
    for child in wandb_dir.iterdir():
        if not is_valid_run_dir(child):
            continue

        run_time = parse_run_time(child)
        if run_time >= cutoff:
            candidates.append((run_time, child))

    candidates.sort(key=lambda x: x[0], reverse=True)

    if not candidates:
        print(f"[INFO] Không có run nào trong {args.days} ngày gần đây để sync.")
        return

    print(f"[INFO] Tìm thấy {len(candidates)} run cần khôi phục:")
    for run_time, run_dir in candidates:
        print(f"  - {run_dir.name}    ({run_time.strftime('%Y-%m-%d %H:%M:%S')})")

    restored = []
    failed = []

    for _, run_dir in candidates:
        print(f"\n===== PROCESSING: {run_dir.name} =====")

        if args.always_new_id:
            code, output, new_id = sync_run_new_id(
                run_dir,
                entity=args.entity,
                project=args.project,
                dry_run=args.dry_run,
            )
            if code == 0:
                restored.append((run_dir.name, f"new-id:{new_id}"))
            else:
                failed.append((run_dir.name, output.strip()[-500:]))
            continue

        code, output = sync_run_original_id(
            run_dir,
            include_synced=args.include_synced,
            entity=args.entity,
            project=args.project,
            dry_run=args.dry_run,
        )

        if code == 0:
            restored.append((run_dir.name, "original-id"))
            continue

        if should_retry_with_new_id(output):
            print("[INFO] ID cũ không dùng lại được, thử sync với ID mới...")
            code2, output2, new_id = sync_run_new_id(
                run_dir,
                entity=args.entity,
                project=args.project,
                dry_run=args.dry_run,
            )
            if code2 == 0:
                restored.append((run_dir.name, f"new-id:{new_id}"))
            else:
                failed.append((run_dir.name, output2.strip()[-500:]))
        else:
            failed.append((run_dir.name, output.strip()[-500:]))

    print("\n===== SUMMARY =====")
    print(f"[OK] Khôi phục được {len(restored)} run")
    for name, mode in restored:
        print(f"  - {name} [{mode}]")

    if failed:
        print(f"\n[WARN] Có {len(failed)} run lỗi:")
        for name, msg in failed:
            print(f"  - {name}")
            if msg:
                print(f"    {msg}")
        sys.exit(2)

    print("\n[OK] Hoàn tất.")


if __name__ == "__main__":
    main()