import random
import string
import argparse
import time
import pprint
import glob
from omegaconf import OmegaConf
import os
import torch

from diffusionsat import SatUNet, DiffusionSatPipeline
from diffusers import AutoencoderKL

from archs.ldm_extractor import LDMExtractor
from utils.lr_scheduler import WarmupCosineDecayScheduler
from utils.utils import load_config, combine_configs, fix_seed
from utils.logger import init_logger
from utils.val_logger import val_log

from datasets.utils import get_datasets
from utils.tasks import create_decoder, set_criterion

import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def save_model(cfg, decoder_net, optimizer, lr_scheduler, epoch, global_step, save_path):
    dict_to_save = {
        "epoch": epoch,
        "global_step": global_step,
        "config": cfg,
        "decoder_net": decoder_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict()
    }
    
    torch.save(dict_to_save, save_path)
   
    
def load_model(cfg, checkpoint_path, decoder_net, optimizer, lr_scheduler):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
    
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
    
    decoder_net.load_state_dict(checkpoint["decoder_net"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    
    epoch = checkpoint["epoch"]
    global_step = checkpoint.get("global_step", 0)
    
    return epoch, global_step
    
         
def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--val_batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--exp_folder", type=str)
    parser.add_argument("--fuser", type=str)

    parser.add_argument("--resume_from", type=str, default=None)

    parser.add_argument("--device", type=str)
    parser.add_argument("--seed", type=int)

    parser.add_argument("--lr", type=float)
    parser.add_argument("--max_train_epochs", type=int)
    parser.add_argument("--wandb_logdir", type=str)
    parser.add_argument("--eval_every_n_steps", type=int)
    parser.add_argument("--report_to", type=str)

    parser.add_argument("--save_timesteps", nargs="+", type=int)
    parser.add_argument("--num_experts", type=int)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--early_stopping_patience", type=int)
    parser.add_argument("--op_weight_decay", type=float)
    parser.add_argument("--limited_label", type=float)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = combine_configs(args, cfg)
    
    fix_seed(cfg.seed)
    
    # Create unique experiment folder with timestamp and random tag
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    random_tag = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    
    exp_name = f"{cfg.dataset_name}-{cfg.fuser}"
    exp_folder = f"{cfg.exp_folder}/{timestamp}-{exp_name}-{random_tag}"
    os.makedirs(exp_folder, exist_ok=True)

    logger_file_path = f"{exp_folder}/log.txt"
    logger = init_logger(logger_file_path)
    logger.info("============ Initialize logger ============")
    logger.info(pprint.pformat(OmegaConf.to_container(cfg), compact=True).strip("{}"))
    OmegaConf.save(cfg, f"{exp_folder}/config.yaml")
    logger.info("The experiment is saved in %s\n" % exp_folder)
    
    
    logger.info(f"============ Initialize Dataset: {cfg.dataset_name} ============")
    dataset_train, dataset_val, dataset_test = get_datasets(cfg)
    
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=cfg.val_batch_size,
        num_workers=cfg.num_workers,
        drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False,
        batch_size=cfg.val_batch_size,
        num_workers=cfg.num_workers,
        drop_last=False
    )
    # Initialize logging backend
    assert cfg.report_to in ["wandb", "tensorboard"]
    if cfg.report_to == "wandb":
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(project=cfg.wandb_project, name=exp_name, config=cfg_dict)
        wandb.run.name = f"{wandb.run.name}_{str(wandb.run.id)}"
        writer = None
    elif cfg.report_to == "tensorboard":
        writer = SummaryWriter(log_dir=f"{exp_folder}/tensorboard")

  
    logger.info(f"Length of train_loader: {len(train_loader)}")
    logger.info(f"Length of val_loader: {len(val_loader)}")
    logger.info(f"Length of test_loader: {len(test_loader)}")
    
    steps_per_epoch = len(train_loader)
    eval_every_n_steps = cfg.get("eval_every_n_steps", None)
    if eval_every_n_steps is None or eval_every_n_steps > steps_per_epoch:
        eval_every_n_steps = steps_per_epoch
    


    logger.info(f"============ Initialize Model ============")
    logger.info("Initializing DiffusionSat")
    cfg.pretrained_model_name_or_path = cfg.pretrained_model_name_or_path + f"/resolution{cfg.img_size}"
    
    unet = SatUNet.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="checkpoint-150000/unet", revision=cfg.revision,
        num_metadata=cfg.num_metadata, use_metadata=cfg.use_metadata, low_cpu_mem_usage=cfg.low_cpu_mem_usage,
    )
    unet.requires_grad_(False)
    unet.to(cfg.device)

    pipe = DiffusionSatPipeline.from_pretrained(cfg.pretrained_model_name_or_path, unet=unet)
    pipe.to(cfg.device)
    
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae", revision=cfg.revision)
    vae.requires_grad_(False)
    vae.to(cfg.device)

    logger.info("Initializing the Feature Extractor")
    ldm_extractor = LDMExtractor(cfg, pipe)

    logger.info("Initializing the Feature Aggregator and the Task Decoder")
    extraction_dims = ldm_extractor.collected_dims

    decoder = create_decoder(cfg, extraction_dims)
    parameter_groups = [{"params": decoder.parameters(), "lr": cfg.lr}]
    decoder.to(cfg.device)

    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(parameter_groups, lr=cfg.lr, weight_decay=cfg.op_weight_decay)
    lr_scheduler = WarmupCosineDecayScheduler(
        optimizer=optimizer,
        warmup_epochs=cfg.get("warmup_epochs"),
        max_lr=cfg.get("lr"),
        min_lr=cfg.get("min_lr"),
        total_epochs=cfg.get("max_train_epochs"),
    )
    
    # Statistics of trainable parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    logger.info(f"Total trainable parameters: {total_params:,}")
    for i, group in enumerate(parameter_groups, 1):
        group_params = sum(p.numel() for p in group["params"])
        logger.info(f"Group {i}: {group_params:,} parameters, learning rate: {group['lr']}")

    # Set loss criterion and random generator for reproducibility
    criterion = set_criterion(cfg.task, cfg.get("ignore_index", -100))
    g = torch.Generator(device=cfg.device).manual_seed(cfg.seed)

    start_epoch = 0
    global_step = 0
    
    if cfg.get("resume_from", None) is not None:
        start_epoch, global_step = load_model(cfg, cfg.resume_from, decoder, optimizer, lr_scheduler)
        logger.info(f"Resuming training from {cfg.resume_from}")
    else:
        logger.info("No checkpoint found. Starting training from scratch.")
        
    # Early stopping configuration
    best_metric = float('-inf')
    patience = cfg.get('early_stopping_patience', float('inf'))
    no_improve = 0
    best_step = global_step
    best_ckpt_path = None
    
    logger.info(f"============ Start Training ============")
    
    early_stop_triggered = False
    for epoch in range(start_epoch, cfg.max_train_epochs):
        if early_stop_triggered:
            break
        if epoch == start_epoch:
            val_log(cfg, decoder, ldm_extractor, val_loader, vae, g, epoch, global_step, logger, val_type="val", writer=writer)
            val_log(cfg, decoder, ldm_extractor, test_loader, vae, g, epoch, global_step, logger, val_type="test", writer=writer)
        logger.info(f"Starting epoch {epoch+1}, learning rate {lr_scheduler.get_lr()}")
        decoder.train()
        start_time = time.time() 
        with tqdm(enumerate(train_loader), total=len(train_loader), leave=False, desc=f"Epoch {epoch + 1}") as pbar:
            for i, batch in pbar:
                optimizer.zero_grad()
                global_step += 1

                images = batch['rgb'].to(cfg.device)
                labels = batch['label'].to(cfg.device)
                filenames = batch['filename']

                # Encode images to latent space
                latents = vae.encode(images).latent_dist.sample(generator=g) * 0.18215
                latents = latents.to(cfg.device)
                
                # Extract features and perform inference
                with torch.inference_mode():
                    feats, _ = ldm_extractor.forward(latents)
                        
                output_shape = (cfg.original_img_size, cfg.original_img_size) if 'seg' in cfg.task else None
                logits, _ = decoder(feats, output_shape=output_shape)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                current_lr = optimizer.param_groups[0]['lr']
                if cfg.report_to == "wandb":
                    wandb.log({"train/loss": loss.item(), "learning_rate": current_lr}, step=global_step)
                if cfg.report_to == "tensorboard":
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("learning_rate", current_lr, global_step)

                # Step-based validation (if enabled and conditions met)
                if global_step % eval_every_n_steps == 0:
                    val_metric = val_log(
                        cfg, decoder, ldm_extractor, val_loader, vae, g, 
                        epoch + 1, global_step=global_step, logger=logger, 
                        val_type="val", writer=writer
                    )
                    
                    if val_metric > best_metric:
                        # Model improved - save new best checkpoint
                        best_metric = val_metric
                        best_step = global_step
                        no_improve = 0
                        
                        # Remove old checkpoints and save new best
                        best_ckpt_path = f"{exp_folder}/best_ckpt_step_{best_step}.pth"
                        old_ckpts = glob.glob(f"{exp_folder}/best_ckpt_*.pth")
                        for old_checkpoint in old_ckpts:
                            os.remove(old_checkpoint)
                            
                        save_model(cfg, decoder, optimizer, lr_scheduler, epoch + 1, global_step, best_ckpt_path)
                        logger.info(f"Saved best checkpoint at step {global_step} (epoch {epoch + 1}) with metric {best_metric:.4f}")
                    else:
                        # No improvement - increment patience counter
                        no_improve += 1
                        logger.info(
                            f"Best metric: {best_metric:.4f} at step {global_step}, "
                            f"steps without improvement: {no_improve * eval_every_n_steps}"
                        )

                    # Early stopping check for step-based evaluation
                    if no_improve >= patience:
                        logger.info(f"Early stopping triggered after {no_improve} steps without improvement.")
                        early_stop_triggered = True
                        break               
        
        lr_scheduler.step()
        epoch_duration = time.time() - start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds")
        logger.info(f"============ Epoch {epoch+1} Done ============")
       
            
    # Final test evaluation with best checkpoint
    if best_ckpt_path and os.path.exists(best_ckpt_path):
        logger.info(f"Loading best checkpoint from step {best_step} for final test evaluation")
        epoch, global_step = load_model(cfg, best_ckpt_path, decoder, optimizer, lr_scheduler)
        test_metric = val_log(
            cfg, decoder, ldm_extractor, test_loader, vae, g, 
            epoch+1, global_step, logger, val_type="test", writer=writer
        )
        logger.info(f"Final test metric: {test_metric:.4f}")
    else:
        logger.warning("No best checkpoint found; skipping final test evaluation")
                    
    if cfg.report_to == "tensorboard":
        writer.close()
    elif cfg.report_to == "wandb":
        wandb.finish()
    
    
if __name__ == "__main__":
    main()