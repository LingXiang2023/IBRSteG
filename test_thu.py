from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.stereo_human_config import ConfigStereoHuman as ConfigStereoHuman
from ibrsteg.gas import GaussianAttributesSteganographer, load_gas_state_dict
from ibrsteg.runtime import project_path, safe_torch_load, set_seed
from lib.gs_utils.image_utils import psnr
from lib.gs_utils.loss_utils import ssim
from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel

try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


ROOT = Path(__file__).resolve().parent
SEQ_PATTERN = re.compile(r"^(s\d+)a(\d+)(?:_process)?$")
pts2render = None


def get_pts2render():
    global pts2render
    if pts2render is None:
        from lib.GaussianRender import pts2render as imported_pts2render

        pts2render = imported_pts2render
    return pts2render


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IBRSteG on processed THU/THumanMV data.")
    parser.add_argument("--config", default="config/stage.yaml")
    parser.add_argument("--data-root", default=None, help="Processed THU dataset root.")
    parser.add_argument("--gps-checkpoint", default=None, help="Frozen GPS-Gaussian+ checkpoint.")
    parser.add_argument("--gas-checkpoint", default=None, help="IBRSteG/GAS checkpoint.")
    parser.add_argument(
        "--test-seqs",
        nargs="+",
        default=None,
        help="THU test sequence names. Names may be bare (s1a4) or full processed paths.",
    )
    parser.add_argument("--views", nargs="+", type=int, default=[2, 3], help="Novel target view ids.")
    parser.add_argument("--num-rounds", type=int, default=1000)
    parser.add_argument(
        "--protocol",
        choices=["random", "per_scene100", "per_subject100"],
        default="random",
        help=(
            "random: sample random cover/secret pairs across --test-seqs. "
            "per_scene100: evaluate each discovered/selected sXaY scene with Y >= --min-action-id "
            "for --per-scene-rounds rounds. "
            "per_subject100: group discovered/selected scenes by subject s, sample scenes with "
            "Y >= --min-action-id, and run --per-subject-rounds rounds per subject."
        ),
    )
    parser.add_argument("--per-scene-rounds", type=int, default=100)
    parser.add_argument("--per-subject-rounds", type=int, default=100)
    parser.add_argument("--min-action-id", type=int, default=4, help="Minimum action id Y in sXaY for per_scene100.")
    parser.add_argument("--output-dir", default="results/thu_test")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-ratio", type=float, default=0.1)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--max-save", type=int, default=10, help="Maximum rounds with saved visualization images.")
    parser.add_argument("--skip-lpips", action="store_true")
    return parser.parse_args()


def default_test_sequences() -> list[str]:
    return ["s1a4", "s1a5", "s1a6", "s2a4", "s3a5"]


def sequence_label(seq: str) -> str:
    label = Path(seq).name
    if label.endswith("_process"):
        label = label[: -len("_process")]
    return label


def parse_sequence_label(seq: str) -> tuple[str, int] | None:
    match = SEQ_PATTERN.match(sequence_label(seq))
    if not match:
        return None
    return match.group(1), int(match.group(2))


def discover_test_sequences(data_root: str | None, min_action_id: int) -> list[str]:
    if not data_root:
        raise ValueError("--protocol per_scene100 requires --data-root so test scenes can be discovered.")
    test_root = Path(data_root) / "test"
    if not test_root.exists():
        raise FileNotFoundError(f"Test root not found: {test_root}")

    discovered = []
    for path in sorted(test_root.iterdir()):
        if not path.is_dir() or not path.name.endswith("_process"):
            continue
        parsed = parse_sequence_label(path.name)
        if parsed is None:
            continue
        _, action_id = parsed
        if action_id >= min_action_id and (path / "img").exists():
            discovered.append(str(path))
    if not discovered:
        raise ValueError(f"No test scenes matching sXaY_process with Y >= {min_action_id} under {test_root}.")
    return discovered


def resolve_test_sequences(args: argparse.Namespace) -> list[str]:
    if args.protocol in {"per_scene100", "per_subject100"}:
        candidates = args.test_seqs if args.test_seqs is not None else discover_test_sequences(args.data_root, args.min_action_id)
        selected = []
        for seq in candidates:
            parsed = parse_sequence_label(seq)
            if parsed is not None and parsed[1] >= args.min_action_id:
                selected.append(seq)
        if not selected:
            raise ValueError(f"No selected test scenes have action id >= {args.min_action_id}.")
        return selected
    return args.test_seqs if args.test_seqs is not None else default_test_sequences()


def build_cfg(args: argparse.Namespace):
    cfg_obj = ConfigStereoHuman()
    cfg_obj.load(project_path(args.config, ROOT))
    cfg = cfg_obj.get_cfg()
    cfg.defrost()
    if args.data_root:
        cfg.dataset.local_data_root = args.data_root
        cfg.dataset.train_data_root = str(Path(args.data_root) / "train")
        cfg.dataset.val_data_root = str(Path(args.data_root) / "val")
    if args.gps_checkpoint:
        cfg.stage1_ckpt = args.gps_checkpoint
    else:
        cfg.stage1_ckpt = project_path(cfg.stage1_ckpt, ROOT)
    if args.gas_checkpoint:
        cfg.restore_ckpt = args.gas_checkpoint
    else:
        cfg.restore_ckpt = project_path(cfg.restore_ckpt, ROOT)
    cfg.dataset.val_novel_id = args.views
    cfg.record.show_path = args.output_dir
    cfg.freeze()
    return cfg


def crop_center(image_tensor: torch.Tensor, ratio: float) -> torch.Tensor:
    if ratio <= 0:
        return image_tensor
    _, _, height, width = image_tensor.shape
    h0 = int(height * ratio)
    h1 = int(height * (1.0 - ratio))
    w0 = int(width * ratio)
    w1 = int(width * (1.0 - ratio))
    return image_tensor[:, :, h0:h1, w0:w1]


def save_image(image_tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_np = image_tensor[0].detach().clamp(0.0, 1.0).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(str(path), image_np[:, :, ::-1])


def sequence_to_phase(data_root: str | None, seq: str) -> str:
    if seq.endswith("_process"):
        return seq
    seq_path = Path(seq)
    if seq_path.is_absolute() or "/" in seq:
        return str(seq_path) + "_process"
    if data_root:
        return str(Path(data_root) / "test" / seq) + "_process"
    return seq + "_process"


class IBRSteGEvaluator:
    def __init__(self, cfg, args: argparse.Namespace):
        self.cfg = cfg
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lpips_model = None
        if LPIPS_AVAILABLE and not args.skip_lpips:
            self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)
            self.lpips_model.eval()
        elif not args.skip_lpips:
            logging.warning("lpips is not installed; LPIPS values will be 0.")

        self.gps_model = RtStereoHumanModel(cfg, with_gs_render=True).to(self.device)
        self.load_gps_checkpoint(cfg.stage1_ckpt)
        for parameter in self.gps_model.parameters():
            parameter.requires_grad = False
        self.gps_model.eval()

        self.gas = GaussianAttributesSteganographer().to(self.device)
        self.load_gas_checkpoint(cfg.restore_ckpt)
        self.gas.eval()
        for parameter in self.gas.parameters():
            parameter.requires_grad = False

        self.seq_names = [sequence_to_phase(args.data_root, seq) for seq in resolve_test_sequences(args)]
        if len(self.seq_names) < 2:
            raise ValueError("At least two test sequences are required to form cover/secret pairs.")
        self.seq_labels = {seq_name: sequence_label(seq_name) for seq_name in self.seq_names}
        self.seq_subjects = {
            seq_name: (parse_sequence_label(seq_name) or (self.seq_labels[seq_name], 0))[0]
            for seq_name in self.seq_names
        }
        self.loaders = {}
        self.iterators = {}
        for index, seq_name in enumerate(self.seq_names):
            dataset = StereoHumanDataset(cfg.dataset, phase=seq_name)
            generator = torch.Generator()
            generator.manual_seed(args.seed + index)
            loader = DataLoader(
                dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                generator=generator,
            )
            self.loaders[seq_name] = loader
            self.iterators[seq_name] = iter(loader)

    def load_gps_checkpoint(self, checkpoint_path: str) -> None:
        logging.info("Loading frozen GPS-Gaussian+ checkpoint from %s", checkpoint_path)
        checkpoint = safe_torch_load(project_path(checkpoint_path, ROOT), map_location=self.device)
        self.gps_model.load_state_dict(checkpoint["network"], strict=True)

    def load_gas_checkpoint(self, checkpoint_path: str) -> None:
        logging.info("Loading GAS checkpoint from %s", checkpoint_path)
        checkpoint = safe_torch_load(project_path(checkpoint_path, ROOT), map_location=self.device)
        load_gas_state_dict(self.gas, checkpoint, strict=True)

    def fetch_data(self, seq_name: str):
        try:
            batch = next(self.iterators[seq_name])
        except StopIteration:
            self.iterators[seq_name] = iter(self.loaders[seq_name])
            batch = next(self.iterators[seq_name])
        for view in ["lmain", "rmain"]:
            for key, value in batch[view].items():
                if isinstance(value, torch.Tensor):
                    batch[view][key] = value.to(self.device)
        return batch

    def compute_lpips(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        if self.lpips_model is None:
            return 0.0
        with torch.no_grad():
            return float(self.lpips_model(pred * 2.0 - 1.0, gt * 2.0 - 1.0).mean().item())

    def compute_pair_metrics(self, pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
        pred = crop_center(pred, self.args.crop_ratio)
        gt = crop_center(gt, self.args.crop_ratio)
        return {
            "psnr": float(psnr(pred, gt).mean().item()),
            "ssim": float(ssim(pred, gt).mean().item()),
            "lpips": self.compute_lpips(pred, gt),
        }

    def metadata_value(self, value):
        if isinstance(value, torch.Tensor):
            return value.flatten()[0].item()
        if isinstance(value, (list, tuple)):
            return value[0]
        return value

    def evaluate_pair(self, round_idx: int, cover_seq: str, secret_seq: str, save_index: int | None = None) -> dict[str, object]:
        cover_batch = self.fetch_data(cover_seq)
        secret_batch = self.fetch_data(secret_seq)
        with torch.no_grad():
            cover_gam, _, _ = self.gps_model(cover_batch, is_train=False)
            secret_gam, _, _ = self.gps_model(secret_batch, is_train=False)
            stego_scene, recovered_scene = self.gas(cover_gam, secret_gam, "test")
            render_fn = get_pts2render()
            stego_render = render_fn(stego_scene, bg_color=self.cfg.dataset.bg_color)
            recovered_render = render_fn(recovered_scene, bg_color=self.cfg.dataset.bg_color)

        stego_pred = stego_render["novel_view"]["img_pred"]
        cover_gt = stego_render["novel_view"]["img"].to(self.device)
        recovered_pred = recovered_render["novel_view"]["img_pred"]
        secret_gt = recovered_render["novel_view"]["img"].to(self.device)

        cover_metrics = self.compute_pair_metrics(stego_pred, cover_gt)
        secret_metrics = self.compute_pair_metrics(recovered_pred, secret_gt)
        row = {
            "round": round_idx,
            "cover_seq": cover_seq,
            "cover_scene": self.seq_labels[cover_seq],
            "cover_subject": self.seq_subjects[cover_seq],
            "cover_sample": self.metadata_value(stego_render["novel_view"].get("sample_name", "")),
            "cover_view": self.metadata_value(stego_render["novel_view"].get("view_id", -1)),
            "secret_seq": secret_seq,
            "secret_scene": self.seq_labels[secret_seq],
            "secret_subject": self.seq_subjects[secret_seq],
            "secret_sample": self.metadata_value(recovered_render["novel_view"].get("sample_name", "")),
            "secret_view": self.metadata_value(recovered_render["novel_view"].get("view_id", -1)),
            "cover_psnr": cover_metrics["psnr"],
            "cover_ssim": cover_metrics["ssim"],
            "cover_lpips": cover_metrics["lpips"],
            "secret_psnr": secret_metrics["psnr"],
            "secret_ssim": secret_metrics["ssim"],
            "secret_lpips": secret_metrics["lpips"],
        }

        if self.args.save_images and save_index is not None and save_index < self.args.max_save:
            save_image(crop_center(stego_pred, self.args.crop_ratio), self.output_dir / "images" / "stego" / f"{save_index:04d}.jpg")
            save_image(crop_center(cover_gt, self.args.crop_ratio), self.output_dir / "images" / "cover_gt" / f"{save_index:04d}.jpg")
            save_image(
                crop_center(recovered_pred, self.args.crop_ratio),
                self.output_dir / "images" / "recovered_secret" / f"{save_index:04d}.jpg",
            )
            save_image(crop_center(secret_gt, self.args.crop_ratio), self.output_dir / "images" / "secret_gt" / f"{save_index:04d}.jpg")
        return row

    def run_random_protocol(self) -> list[dict[str, object]]:
        rows = []
        for round_idx in tqdm(range(self.args.num_rounds), desc="Testing IBRSteG"):
            cover_seq, secret_seq = random.sample(self.seq_names, 2)
            rows.append(self.evaluate_pair(round_idx, cover_seq, secret_seq, save_index=round_idx))
        return rows

    def run_per_scene_protocol(self) -> list[dict[str, object]]:
        rows = []
        total_rounds = len(self.seq_names) * self.args.per_scene_rounds
        progress = tqdm(total=total_rounds, desc="Testing IBRSteG per scene")
        try:
            for cover_seq in self.seq_names:
                secret_pool = [seq_name for seq_name in self.seq_names if seq_name != cover_seq]
                for scene_round in range(self.args.per_scene_rounds):
                    secret_seq = random.choice(secret_pool)
                    row = self.evaluate_pair(len(rows), cover_seq, secret_seq, save_index=len(rows))
                    row["scene_round"] = scene_round
                    rows.append(row)
                    progress.update(1)
        finally:
            progress.close()
        return rows

    def run_per_subject_protocol(self) -> list[dict[str, object]]:
        rows = []
        subject_to_sequences = defaultdict(list)
        for seq_name in self.seq_names:
            subject_to_sequences[self.seq_subjects[seq_name]].append(seq_name)

        total_rounds = len(subject_to_sequences) * self.args.per_subject_rounds
        progress = tqdm(total=total_rounds, desc="Testing IBRSteG per subject")
        try:
            for subject in sorted(subject_to_sequences):
                cover_pool = subject_to_sequences[subject]
                for subject_round in range(self.args.per_subject_rounds):
                    cover_seq = random.choice(cover_pool)
                    secret_pool = [seq_name for seq_name in self.seq_names if seq_name != cover_seq]
                    secret_seq = random.choice(secret_pool)
                    row = self.evaluate_pair(len(rows), cover_seq, secret_seq, save_index=len(rows))
                    row["subject_round"] = subject_round
                    rows.append(row)
                    progress.update(1)
        finally:
            progress.close()
        return rows

    def run(self) -> dict[str, object]:
        if self.args.protocol == "per_scene100":
            rows = self.run_per_scene_protocol()
        elif self.args.protocol == "per_subject100":
            rows = self.run_per_subject_protocol()
        else:
            rows = self.run_random_protocol()
        summary = self.summarize(rows)
        scene_summary = self.summarize_grouped(rows, "cover_scene")
        subject_summary = self.summarize_grouped(rows, "cover_subject")
        scene_mean_summary = self.summarize_summary_rows(scene_summary)
        subject_mean_summary = self.summarize_summary_rows(subject_summary)
        self.write_outputs(rows, summary, scene_summary, subject_summary, scene_mean_summary, subject_mean_summary)
        return {
            "rows": rows,
            "summary": summary,
            "scene_summary": scene_summary,
            "subject_summary": subject_summary,
            "scene_mean_summary": scene_mean_summary,
            "subject_mean_summary": subject_mean_summary,
        }

    def summarize(self, rows: list[dict[str, float]]) -> dict[str, float]:
        metric_names = ["cover_psnr", "cover_ssim", "cover_lpips", "secret_psnr", "secret_ssim", "secret_lpips"]
        summary = {"num_rounds": len(rows)}
        for name in metric_names:
            values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
            summary[f"{name}_mean"] = float(values.mean())
            summary[f"{name}_std"] = float(values.std())
        return summary

    def summarize_grouped(self, rows: list[dict], group_key: str) -> list[dict[str, object]]:
        grouped = defaultdict(list)
        for row in rows:
            grouped[row[group_key]].append(row)
        summaries = []
        for key in sorted(grouped):
            summary = self.summarize(grouped[key])
            summary[group_key] = key
            summaries.append(summary)
        return summaries

    def summarize_summary_rows(self, summaries: list[dict[str, object]]) -> dict[str, float]:
        metric_names = ["cover_psnr", "cover_ssim", "cover_lpips", "secret_psnr", "secret_ssim", "secret_lpips"]
        out = {"num_groups": len(summaries)}
        if not summaries:
            return out
        for name in metric_names:
            values = np.asarray([float(row[f"{name}_mean"]) for row in summaries], dtype=np.float64)
            out[f"{name}_mean"] = float(values.mean())
            out[f"{name}_std"] = float(values.std())
        return out

    def write_summary_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def write_outputs(
        self,
        rows: list[dict],
        summary: dict[str, float],
        scene_summary: list[dict[str, object]],
        subject_summary: list[dict[str, object]],
        scene_mean_summary: dict[str, float],
        subject_mean_summary: dict[str, float],
    ) -> None:
        csv_path = self.output_dir / "metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        self.write_summary_csv(self.output_dir / "scene_metrics.csv", scene_summary)
        self.write_summary_csv(self.output_dir / "subject_metrics.csv", subject_summary)
        payload = {
            "protocol": self.args.protocol,
            "gps_checkpoint": self.cfg.stage1_ckpt,
            "gas_checkpoint": self.cfg.restore_ckpt,
            "test_sequences": self.seq_names,
            "views": self.args.views,
            "summary": summary,
            "scene_summary": scene_summary,
            "subject_summary": subject_summary,
            "scene_mean_summary": scene_mean_summary,
            "subject_mean_summary": subject_mean_summary,
            "rows": rows,
        }
        (self.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s")
    set_seed(args.seed)
    cfg = build_cfg(args)
    evaluator = IBRSteGEvaluator(cfg, args)
    result = evaluator.run()
    summary = result["summary"]
    logging.info(
        "Cover PSNR/SSIM/LPIPS: %.4f / %.4f / %.4f",
        summary["cover_psnr_mean"],
        summary["cover_ssim_mean"],
        summary["cover_lpips_mean"],
    )
    logging.info(
        "Secret PSNR/SSIM/LPIPS: %.4f / %.4f / %.4f",
        summary["secret_psnr_mean"],
        summary["secret_ssim_mean"],
        summary["secret_lpips_mean"],
    )


if __name__ == "__main__":
    main()
