import lerobot_patches.custom_patches  # Ensure custom patches are applied, DON'T REMOVE THIS LINE!
from lerobot.configs.policies import PolicyFeature
from typing import Any
import math
import os

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from pathlib import Path
from functools import partial

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
from hydra.utils import instantiate
from diffusers.optimization import get_scheduler

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.random_utils import set_seed

from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from kuavo_train.wrapper.dataset.LeRobotDatasetWrapper import CustomLeRobotDataset
from kuavo_train.utils.augmenter import crop_image, resize_image, DeterministicAugmenterColor
from kuavo_train.utils.utils import save_rng_state, load_rng_state
from lerobot.policies.act.modeling_act import ACTPolicy
from diffusers.optimization import get_scheduler
from utils.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig

from functools import partial
from contextlib import nullcontext


def build_augmenter(cfg):
    """Since operations such as cropping and resizing in LeRobot are implemented at the model level 
    rather than at the data level, we provide only RGB image augmentations on the data side here, 
    with support for customization. For more details, refer to configs/policy/diffusion_config.yaml. 
    To define custom transformations, please see utils.transforms.py."""

    img_tf_cfg = ImageTransformsConfig(
        enable=cfg.get("enable", False),
        max_num_transforms=cfg.get("max_num_transforms", 3),
        random_order=cfg.get("random_order", False),
        tfs={}
    )

    # deal tfs part
    if "tfs" in cfg:
        for name, tf_dict in cfg["tfs"].items():
            img_tf_cfg.tfs[name] = ImageTransformConfig(
                weight=tf_dict.get("weight", 1.0),
                type=tf_dict.get("type", "Identity"),
                kwargs=tf_dict.get("kwargs", {}),
            )
    return ImageTransforms(img_tf_cfg)


def build_delta_timestamps(dataset_metadata, policy_cfg):
    """Build delta timestamps for observations and actions."""
    obs_indices = getattr(policy_cfg, "observation_delta_indices", None)
    act_indices = getattr(policy_cfg, "action_delta_indices", None)
    if obs_indices is None and act_indices is None:
        return None

    delta_timestamps = {}
    for key in dataset_metadata.info["features"]:
        if "observation" in key and obs_indices is not None:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in obs_indices]
        elif "action" in key and act_indices is not None:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in act_indices]

    return delta_timestamps if delta_timestamps else None


def build_optimizer_and_scheduler(policy, cfg, total_frames, world_size: int):
    """Return optimizer and scheduler."""
    optimizer = policy.config.get_optimizer_preset().build(policy.parameters())
    # If `max_training_step` is specified, it takes precedence;
    # otherwise, the value is automatically determined based on `max_epoch`.
    if cfg.training.max_training_step is None:
        effective_batch = cfg.training.batch_size * cfg.training.accumulation_steps * world_size
        updates_per_epoch = math.ceil(total_frames / max(effective_batch, 1))
        num_training_steps = cfg.training.max_epoch * updates_per_epoch
    else:
        num_training_steps = cfg.training.max_training_step
    lr_scheduler = policy.config.get_scheduler_preset()
    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler.build(optimizer, num_training_steps)
    else:
        lr_scheduler = get_scheduler(
            name=cfg.training.scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.scheduler_warmup_steps,
            num_training_steps=num_training_steps,
        )

    # or you can set your optimizer and lr_scheduler here and replace it.
    return optimizer, lr_scheduler


def build_policy_config(cfg, input_features, output_features):
    def _normalize_feature_dict(d: Any) -> dict[str, PolicyFeature]:
        if isinstance(d, DictConfig):
            d = OmegaConf.to_container(d, resolve=True)
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict or DictConfig, got {type(d)}")

        return {
            k: PolicyFeature(**v) if isinstance(v, dict) and not isinstance(v, PolicyFeature) else v
            for k, v in d.items()
        }

    policy_cfg = instantiate(
        cfg.policy,
        input_features=input_features,
        output_features=output_features,
        device=cfg.training.device,
    )
                
    policy_cfg.input_features = _normalize_feature_dict(policy_cfg.input_features)
    policy_cfg.output_features = _normalize_feature_dict(policy_cfg.output_features)
    return policy_cfg

def build_policy(name, policy_cfg, dataset_stats):
    policy = {
        "diffusion": CustomDiffusionPolicyWrapper,
        "act": ACTPolicy,
    }[name](policy_cfg, dataset_stats)
    return policy

def build_policy_config(cfg, input_features, output_features):
    def _normalize_feature_dict(d: Any) -> dict[str, PolicyFeature]:
        if isinstance(d, DictConfig):
            d = OmegaConf.to_container(d, resolve=True)
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict or DictConfig, got {type(d)}")

        return {
            k: PolicyFeature(**v) if isinstance(v, dict) and not isinstance(v, PolicyFeature) else v
            for k, v in d.items()
        }

    policy_cfg = instantiate(
        cfg.policy,
        input_features=input_features,
        output_features=output_features,
        device=cfg.training.device,
    )
                
    policy_cfg.input_features = _normalize_feature_dict(policy_cfg.input_features)
    policy_cfg.output_features = _normalize_feature_dict(policy_cfg.output_features)
    return policy_cfg




@hydra.main(config_path="../configs/policy/", config_name="act_config", version_base=None)
def main(cfg: DictConfig):
    distributed = dist.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0")) if distributed else 0
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) if distributed else 0

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
        cfg.training.device = str(device)

    set_seed(cfg.training.seed)

    # Setup output directory
    output_directory = Path(cfg.training.output_directory) / f"run_{cfg.timestamp}"
    if rank == 0:
        output_directory.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    writer = SummaryWriter(log_dir=str(output_directory)) if rank == 0 else None

    # Dataset metadata and features
    repoid = cfg.repoid
    if isinstance(repoid, str):
        # Allow passing a CLI string such as "['lerobot1-200','lerobot201-400']"
        # to make torchrun parameter forwarding easier in SLURM scripts.
        repoid = OmegaConf.create({"repoid": repoid}).repoid

    dataset_metadata = LeRobotDatasetMetadata(repoid, root=cfg.root)
    print("camera_keys:", dataset_metadata.camera_keys)
    print("Original dataset features:", dataset_metadata.features)

    features = dataset_to_policy_features(dataset_metadata.features)
    input_features = {k: ft for k, ft in features.items() if ft.type is not FeatureType.ACTION}
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}

    print(f"Input features: {input_features}")
    print(f"Output features: {output_features}")

    # instantiate the policy
    policy_cfg = build_policy_config(cfg, input_features, output_features)
    print("policy_cfg", policy_cfg)

    # Build policy
    world_size = dist.get_world_size() if distributed else 1
    total_batch_size = cfg.training.batch_size
    per_device_batch_size = max(1, math.ceil(total_batch_size / world_size))
    cfg.training.batch_size = per_device_batch_size
    if rank == 0 and distributed:
        print(f"Total batch size: {total_batch_size}, per-device batch size: {per_device_batch_size}, world size: {world_size}")

    policy = build_policy(cfg.policy_name, policy_cfg, dataset_stats=dataset_metadata.stats)
    optimizer, lr_scheduler = build_optimizer_and_scheduler(policy, cfg, dataset_metadata.info["total_frames"], world_size)
    
    # Initialize AMP GradScaler if use_amp is True
    amp_requested = bool(getattr(cfg.policy, "use_amp", False))
    amp_enabled = amp_requested and device.type == "cuda"

    # autocast context (cuda, or no-op when disabled/non-cuda)
    has_torch_autocast = hasattr(torch, "autocast")
    def make_autocast(enabled: bool):
        if not enabled:
            return nullcontext()
        if device.type == "cuda":
            if has_torch_autocast:
                return torch.autocast(device_type="cuda")
            else:
                from torch.cuda.amp import autocast as cuda_autocast  # noqa
                return cuda_autocast()
        # Fallback: disable on non-cuda to avoid dtype surprises
        return nullcontext()

    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(device=device.type, enabled=amp_enabled)
    # print("scaler", device.type, make_autocast(amp_enabled))
    # Initialize training state variables
    start_epoch = 0
    steps = 0
    best_loss = float('inf')

    # ===== Resume logic (perfect resume for AMP & RNG) =====

    if cfg.training.resume and cfg.training.resume_timestamp:
        resume_path = Path(cfg.training.output_directory) / cfg.training.resume_timestamp
        if rank == 0:
            print("Resuming from:", resume_path)
        try:
            checkpoint = None
            if rank == 0:
                # Load RNG state
                load_rng_state(resume_path / "rng_state.pth")

                # Load policy
                policy = policy.from_pretrained(resume_path, strict=True)

                """ Warning: using `from_pretrained` creates a new policy instance,
                so the optimizer must be reinitialized here! """
                optimizer, lr_scheduler = build_optimizer_and_scheduler(policy, cfg, dataset_metadata.info["total_frames"], world_size)

                # Load optimizer, scheduler, scaler and training state
                checkpoint = torch.load(resume_path / "learning_state.pth", map_location=device)
            if distributed:
                dist.barrier()

            if distributed:
                state_payload = [
                    policy.state_dict() if rank == 0 else None,
                    optimizer.state_dict() if rank == 0 else None,
                    lr_scheduler.state_dict() if rank == 0 else None,
                    checkpoint if rank == 0 else None,
                ]
                dist.broadcast_object_list(state_payload, src=0)
                state_dict, optimizer_state, lr_scheduler_state, checkpoint = state_payload
                policy.load_state_dict(state_dict)
                optimizer.load_state_dict(optimizer_state)
                if lr_scheduler_state is not None:
                    lr_scheduler.load_state_dict(lr_scheduler_state)
            else:
                optimizer.load_state_dict(checkpoint["optimizer"])
                if "lr_scheduler" in checkpoint:
                    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

            if checkpoint is not None:
                if "scaler" in checkpoint and amp_enabled:
                    scaler.load_state_dict(checkpoint["scaler"])

                if "steps" in checkpoint:
                    steps = checkpoint["steps"]

                if "epoch" in checkpoint:
                    start_epoch = checkpoint["epoch"]

                if "best_loss" in checkpoint:
                    best_loss = checkpoint["best_loss"]

                if rank == 0:
                    for file in resume_path.glob("events.*"):
                        shutil.copy(file, output_directory)

            if distributed:
                dist.barrier()

            if rank == 0:
                print(f"Resumed training from epoch {start_epoch}, step {steps}")
        except Exception as e:
            if rank == 0:
                print("Failed to load checkpoint:", e)
            return
    else:
        if rank == 0:
            print("Training from scratch!")

    policy.to(device)
    policy.train()
    if distributed:
        policy = DDP(
            policy,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=getattr(cfg.training, "find_unused_parameters", False),
        )
    if rank == 0:
        print(f"Total parameters: {sum(p.numel() for p in policy.parameters()):,}")
        print(f"Using AMP: {amp_enabled}")
    # Build dataset and dataloader
    delta_timestamps = build_delta_timestamps(dataset_metadata, policy_cfg)

    image_transforms = build_augmenter(cfg.training.RGB_Augmenter)
    dataset = LeRobotDataset(
        repoid,
        delta_timestamps=delta_timestamps,
        root=cfg.root,
        image_transforms=image_transforms,
    )

    # Training loop
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=cfg.training.drop_last,
        )

    dataloader = DataLoader(
        dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        sampler=sampler,
        pin_memory=(device.type != "cpu"),
        drop_last=cfg.training.drop_last,
        prefetch_factor=1,
    )

    for epoch in range(start_epoch, cfg.training.max_epoch):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{cfg.training.max_epoch}",
            disable=rank != 0,
        )

        total_loss = 0.0
        for batch in epoch_bar:
            
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            with make_autocast(amp_enabled):
                loss, _ = policy.forward(batch)
            # Scale loss and backward with AMP if enabled
            scaled_loss = loss / cfg.training.accumulation_steps
            
            if amp_enabled:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (steps + 1) % cfg.training.accumulation_steps == 0:
                if amp_enabled:
                    # Optionally unscale and clip gradients here if you use clipping
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step()

            if writer and steps % cfg.training.log_freq == 0:
                writer.add_scalar("train/loss", scaled_loss.item(), steps)
                if lr_scheduler is not None:
                    writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], steps)
                epoch_bar.set_postfix(
                    loss=f"{scaled_loss.item():.3f}",
                    step=steps,
                    lr=lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else None,
                )

            steps += 1
            total_loss += scaled_loss.item()

        # Update best loss
        if rank == 0:
            if total_loss < best_loss:
                best_loss = total_loss
                # Save best model
                (policy.module if isinstance(policy, DDP) else policy).save_pretrained(output_directory / "best")
            # Save checkpoint every N epochs
            if (epoch + 1) % cfg.training.save_freq_epoch == 0:
                (policy.module if isinstance(policy, DDP) else policy).save_pretrained(output_directory / f"epoch{epoch+1}")

            # Save last checkpoint (includes AMP scaler & progress for perfect resume)
            (policy.module if isinstance(policy, DDP) else policy).save_pretrained(output_directory)
            # Save training state including optimizer, scheduler, scaler, and step/epoch info
            checkpoint = {
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                "scaler": scaler.state_dict() if amp_enabled else None,
                "steps": steps,
                "epoch": epoch + 1,
                "best_loss": best_loss
            }
            torch.save(checkpoint, output_directory / "learning_state.pth")
            save_rng_state(output_directory / "rng_state.pth")

        if distributed:
            dist.barrier()

    if writer:
        writer.close()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
