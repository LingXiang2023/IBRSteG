from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.stereo_human_config import ConfigStereoHuman as ConfigStereoHuman
from ibrsteg.gas import GaussianAttributesSteganographer, load_gas_state_dict
from ibrsteg.losses import attribute_loss
from ibrsteg.runtime import project_path, safe_torch_load, set_seed
from lib.gs_utils.image_utils import psnr
from lib.gs_utils.loss_utils import l1_loss, ssim
from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from lib.train_recoder import Logger, file_backup

try:
    from pytorch3d.loss import chamfer_distance

    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False


ROOT = Path(__file__).resolve().parent
pts2render = None


def get_pts2render():
    global pts2render
    if pts2render is None:
        from lib.GaussianRender import pts2render as imported_pts2render

        pts2render = imported_pts2render
    return pts2render


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IBRSteG/GAS on the THU THumanMV-style dataset.")
    parser.add_argument("--config", default="config/stage.yaml", help="YACS config file.")
    parser.add_argument("--data-root", default=None, help="Processed THU dataset root.")
    parser.add_argument("--train-data-root", default=None, help="Processed train split root.")
    parser.add_argument("--val-data-root", default=None, help="Processed val split root.")
    parser.add_argument("--gps-checkpoint", default=None, help="Frozen GPS-Gaussian+ checkpoint.")
    parser.add_argument("--resume", default=None, help="Resume GAS checkpoint.")
    parser.add_argument("--output-dir", default="experiments/ibrsteg_thu", help="Experiment output directory.")
    parser.add_argument("--experiment-name", default=None, help="Optional experiment name.")
    parser.add_argument("--num-steps", type=int, default=None, help="Override training steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override GAS learning rate.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--no-chamfer", action="store_true", help="Disable Chamfer regularization.")
    parser.add_argument("--save-source-snapshot", action="store_true", help="Backup source files into the experiment folder.")
    return parser.parse_args()


def build_cfg(args: argparse.Namespace):
    cfg_obj = ConfigStereoHuman()
    cfg_obj.load(project_path(args.config, ROOT))
    cfg = cfg_obj.get_cfg()
    cfg.defrost()

    if args.data_root:
        cfg.dataset.local_data_root = args.data_root
        cfg.dataset.train_data_root = str(Path(args.data_root) / "train")
        cfg.dataset.val_data_root = str(Path(args.data_root) / "val")
    if args.train_data_root:
        cfg.dataset.train_data_root = args.train_data_root
    if args.val_data_root:
        cfg.dataset.val_data_root = args.val_data_root
    if args.gps_checkpoint:
        cfg.stage1_ckpt = args.gps_checkpoint
    else:
        cfg.stage1_ckpt = project_path(cfg.stage1_ckpt, ROOT)
    cfg.restore_ckpt = args.resume
    if args.num_steps is not None:
        cfg.num_steps = args.num_steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr

    experiment_name = args.experiment_name or f"ibrsteg_{datetime.now().strftime('%m%d_%H%M')}"
    cfg.name = experiment_name
    cfg.exp_name = experiment_name
    output_dir = Path(args.output_dir)
    cfg.record.ckpt_path = str(output_dir / "ckpt")
    cfg.record.show_path = str(output_dir / "show")
    cfg.record.logs_path = str(output_dir / "logs")
    cfg.record.file_path = str(output_dir / "file")
    cfg.freeze()
    return cfg


class IBRSteGTrainer:
    def __init__(self, cfg, args: argparse.Namespace):
        self.cfg = cfg
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = cfg.batch_size
        self.use_chamfer = (not args.no_chamfer) and PYTORCH3D_AVAILABLE
        if not self.use_chamfer and not args.no_chamfer:
            logging.warning("pytorch3d is unavailable; Chamfer regularization is disabled.")

        self.gps_model = RtStereoHumanModel(cfg, with_gs_render=True).to(self.device)
        self.load_gps_checkpoint(cfg.stage1_ckpt)
        for parameter in self.gps_model.parameters():
            parameter.requires_grad = False
        self.gps_model.eval()

        self.gas = GaussianAttributesSteganographer().to(self.device)
        self.train_set = StereoHumanDataset(cfg.dataset, phase="train")
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        self.train_iterator = iter(self.train_loader)
        self.val_set = StereoHumanDataset(cfg.dataset, phase="val")
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=1,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        self.len_val = max(1, int(len(self.val_loader) / self.val_set.val_boost))
        self.val_iterator = iter(self.val_loader)

        self.optimizer = optim.Adam(self.gas.parameters(), lr=cfg.lr, weight_decay=1e-5, betas=(0.5, 0.999))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.1)
        self.logger = Logger(self.scheduler, cfg.record)
        self.scaler = GradScaler(enabled=cfg.raft.mixed_precision)
        self.total_steps = 0

        if cfg.restore_ckpt:
            self.load_gas_checkpoint(cfg.restore_ckpt)

    def fetch_data(self, phase: str):
        iterator_name = f"{phase}_iterator"
        loader_name = f"{phase}_loader"
        try:
            batch = next(getattr(self, iterator_name))
        except StopIteration:
            setattr(self, iterator_name, iter(getattr(self, loader_name)))
            batch = next(getattr(self, iterator_name))
        except Exception:
            setattr(self, iterator_name, iter(getattr(self, loader_name)))
            batch = next(getattr(self, iterator_name))

        for view in ["lmain", "rmain"]:
            for key, value in batch[view].items():
                if isinstance(value, torch.Tensor):
                    batch[view][key] = value.to(self.device)
        return batch

    def load_gps_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint_path = project_path(checkpoint_path, ROOT)
        logging.info("Loading frozen GPS-Gaussian+ checkpoint from %s", checkpoint_path)
        checkpoint = safe_torch_load(checkpoint_path, map_location=self.device)
        self.gps_model.load_state_dict(checkpoint["network"], strict=True)

    def load_gas_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> None:
        checkpoint_path = project_path(checkpoint_path, ROOT)
        logging.info("Loading GAS checkpoint from %s", checkpoint_path)
        checkpoint = safe_torch_load(checkpoint_path, map_location=self.device)
        load_gas_state_dict(self.gas, checkpoint, strict=True)
        if load_optimizer and "optimizer" in checkpoint:
            self.total_steps = int(checkpoint.get("total_steps", -1)) + 1
            self.logger.total_steps = self.total_steps
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])

    def save_checkpoint(self, path: Path, show_log: bool = True) -> None:
        if show_log:
            logging.info("Saving checkpoint to %s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = self.gas.state_dict()
        torch.save(
            {
                "total_steps": self.total_steps,
                "gas": state_dict,
                "conet": state_dict,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            path,
        )

    def compute_chamfer(self, rendered_items) -> torch.Tensor:
        if not self.use_chamfer:
            return torch.zeros([], device=self.device)
        chamfer = torch.zeros([], device=self.device)
        for item in rendered_items:
            left_xyz = item["lmain"]["xyz"]
            right_xyz = item["rmain"]["xyz"]
            for batch_idx in range(self.batch_size):
                left_valid = item["lmain"]["pts_valid"][batch_idx, :]
                right_valid = item["rmain"]["pts_valid"][batch_idx, :]
                left_points = left_xyz[batch_idx, :, :][left_valid].view(1, -1, 3).contiguous()
                right_points = right_xyz[batch_idx, :, :][right_valid].view(1, -1, 3).contiguous()
                sample_count = min(left_points.shape[1], right_points.shape[1], 10000)
                if sample_count <= 0:
                    continue
                left_idx = np.random.choice(left_points.shape[1], sample_count, replace=False)
                right_idx = np.random.choice(right_points.shape[1], sample_count, replace=False)
                chamfer_i, _ = chamfer_distance(left_points[:, left_idx], right_points[:, right_idx])
                chamfer = chamfer + chamfer_i
        return chamfer / max(1, self.batch_size)

    def train(self) -> None:
        rolling = []
        for _ in tqdm(range(self.total_steps, self.cfg.num_steps), desc="Training IBRSteG"):
            self.optimizer.zero_grad(set_to_none=True)
            self.gas.train()

            cover_batch = self.fetch_data("train")
            secret_batch = self.fetch_data("train")
            with torch.no_grad():
                cover_gam, _, _ = self.gps_model(cover_batch)
                secret_gam, _, _ = self.gps_model(secret_batch)

            stego_scene, recovered_scene, gam_losses = self.gas(cover_gam, secret_gam, "train")
            render_fn = get_pts2render()
            stego_render = render_fn(stego_scene, bg_color=self.cfg.dataset.bg_color)
            recovered_render = render_fn(recovered_scene, bg_color=self.cfg.dataset.bg_color)

            stego_pred = stego_render["novel_view"]["img_pred"]
            cover_gt = stego_render["novel_view"]["img"].to(self.device)
            recovered_pred = recovered_render["novel_view"]["img_pred"]
            secret_gt = recovered_render["novel_view"]["img"].to(self.device)

            l1 = l1_loss(stego_pred, cover_gt) + l1_loss(recovered_pred, secret_gt)
            ssim_loss = (1.0 - ssim(stego_pred, cover_gt)) + (1.0 - ssim(recovered_pred, secret_gt))
            chamfer = self.compute_chamfer([stego_render, recovered_render])
            gam_loss = attribute_loss(gam_losses, self.cfg.loss)
            loss = 0.8 * l1 + 0.2 * ssim_loss + 0.5 * chamfer + gam_loss

            rolling.append(float(loss.item()))
            if self.total_steps and self.total_steps % self.cfg.record.show_freq == 0:
                logging.info("step %d loss %.6f", self.total_steps, sum(rolling) / len(rolling))
                rolling.clear()

            metrics = {
                "l1": float(l1.item()),
                "ssim": float(ssim_loss.item()),
                "chamfer": float(chamfer.item()),
                "gam": float(gam_loss.item()),
            }
            self.logger.push(metrics)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.gas.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scheduler.step()
            self.scaler.update()

            if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                self.logger.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], self.total_steps)
                self.save_checkpoint(Path(self.cfg.record.ckpt_path) / f"{self.cfg.name}_latest.pth", show_log=False)

            if self.total_steps and self.total_steps % self.cfg.record.eval_freq == 0:
                self.gas.eval()
                self.run_eval()
                self.gas.train()

            if self.total_steps in self.cfg.record.save_iter:
                self.save_checkpoint(Path(self.cfg.record.ckpt_path) / f"iter{self.total_steps}.pth")

            self.total_steps += 1

        self.logger.close()
        self.save_checkpoint(Path(self.cfg.record.ckpt_path) / f"{self.cfg.name}_final.pth")

    def run_eval(self) -> None:
        logging.info("Running validation ...")
        psnr_values = []
        show_idx = set(np.random.choice(list(range(self.len_val)), 1))
        for idx in range(self.len_val):
            cover_batch = self.fetch_data("val")
            secret_batch = self.fetch_data("val")
            with torch.no_grad():
                cover_gam, _, _ = self.gps_model(cover_batch, is_train=False)
                secret_gam, _, _ = self.gps_model(secret_batch, is_train=False)
                stego_scene, recovered_scene = self.gas(cover_gam, secret_gam, "test")
                render_fn = get_pts2render()
                stego_render = render_fn(stego_scene, bg_color=self.cfg.dataset.bg_color)
                recovered_render = render_fn(recovered_scene, bg_color=self.cfg.dataset.bg_color)

                stego_psnr = psnr(
                    stego_render["novel_view"]["img_pred"],
                    stego_render["novel_view"]["img"].to(self.device),
                ).mean().double()
                recovered_psnr = psnr(
                    recovered_render["novel_view"]["img_pred"],
                    recovered_render["novel_view"]["img"].to(self.device),
                ).mean().double()
                psnr_values.append(float(((stego_psnr + recovered_psnr) / 2).item()))

                if idx in show_idx:
                    self.save_eval_image(stego_render["novel_view"]["img_pred"], "stego")
                    self.save_eval_image(recovered_render["novel_view"]["img_pred"], "recovered")

        val_psnr = float(np.round(np.mean(np.asarray(psnr_values)), 4))
        logging.info("Validation PSNR at step %d: %.4f", self.total_steps, val_psnr)
        self.logger.write_dict({"val_psnr": val_psnr}, write_step=self.total_steps)

    def save_eval_image(self, image: torch.Tensor, suffix: str) -> None:
        image_np = image[0].detach().clamp(0.0, 1.0).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        output_path = Path(self.cfg.record.show_path) / f"{self.total_steps}_{suffix}.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image_np[:, :, ::-1])


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s")
    set_seed(args.seed)
    cfg = build_cfg(args)
    for path in [cfg.record.ckpt_path, cfg.record.show_path, cfg.record.logs_path, cfg.record.file_path]:
        Path(path).mkdir(parents=True, exist_ok=True)
    if args.save_source_snapshot:
        file_backup(cfg.record.file_path, cfg, train_script=Path(__file__).name)
    trainer = IBRSteGTrainer(cfg, args)
    trainer.train()


if __name__ == "__main__":
    main()
