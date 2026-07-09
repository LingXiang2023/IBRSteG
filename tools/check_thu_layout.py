from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_TEST_SEQS = ["s1a4", "s1a5", "s1a6", "s2a4", "s3a5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the processed THU/THumanMV layout used by train_thu.py and test_thu.py.")
    parser.add_argument("--data-root", type=Path, default=Path("data/thu_processed"), help="processed THU root")
    parser.add_argument("--test-seqs", nargs="+", default=DEFAULT_TEST_SEQS, help="bare test names such as s1a4, or *_process names")
    parser.add_argument("--train-views", nargs="+", type=int, default=[2, 3, 4, 5], help="novel views expected for train samples")
    parser.add_argument("--val-views", nargs="+", type=int, default=[2, 3], help="novel views expected for val/test samples")
    parser.add_argument("--max-samples", type=int, default=5, help="samples checked per split/scene; use 0 to scan all")
    return parser.parse_args()


def limited_samples(path: Path, max_samples: int) -> list[Path]:
    img_root = path / "img"
    if not img_root.exists():
        raise FileNotFoundError(f"missing image directory: {img_root}")
    samples = sorted(item for item in img_root.iterdir() if item.is_dir())
    if not samples:
        raise FileNotFoundError(f"no sample directories under: {img_root}")
    if max_samples > 0:
        samples = samples[:max_samples]
    return samples


def require_file(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing: {path}")


def check_sample(split_root: Path, sample_name: str, novel_views: list[int], errors: list[str]) -> None:
    sample_img = split_root / "img" / sample_name
    sample_mask = split_root / "mask" / sample_name
    sample_param = split_root / "parameter" / sample_name

    require_file(sample_img / "0.jpg", errors)
    require_file(sample_img / "1.jpg", errors)
    require_file(sample_mask / "0.jpg", errors)
    require_file(sample_mask / "1.jpg", errors)
    require_file(sample_param / "0_1.json", errors)
    for view_id in novel_views:
        require_file(sample_img / f"{view_id}.jpg", errors)
        require_file(sample_param / f"{view_id}_intrinsic.npy", errors)
        require_file(sample_param / f"{view_id}_extrinsic.npy", errors)


def check_split(split_root: Path, novel_views: list[int], max_samples: int, errors: list[str]) -> int:
    samples = limited_samples(split_root, max_samples)
    for sample in samples:
        check_sample(split_root, sample.name, novel_views, errors)
    return len(samples)


def test_seq_root(data_root: Path, seq: str) -> Path:
    seq_path = Path(seq)
    if seq_path.is_absolute() or "/" in seq:
        return seq_path
    name = seq if seq.endswith("_process") else f"{seq}_process"
    return data_root / "test" / name


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    errors: list[str] = []
    checked: list[str] = []

    for split, views in [("train", args.train_views), ("val", args.val_views)]:
        split_root = data_root / split
        if split_root.exists():
            try:
                count = check_split(split_root, views, args.max_samples, errors)
                checked.append(f"{split}:{count}")
            except FileNotFoundError as exc:
                errors.append(str(exc))
        else:
            errors.append(f"missing split directory: {split_root}")

    for seq in args.test_seqs:
        split_root = test_seq_root(data_root, seq)
        try:
            count = check_split(split_root, args.val_views, args.max_samples, errors)
            checked.append(f"{split_root.name}:{count}")
        except FileNotFoundError as exc:
            errors.append(str(exc))

    if errors:
        print("THU layout check failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("THU layout check passed: " + ", ".join(checked))


if __name__ == "__main__":
    main()
