import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from PIL import Image
import wandb
from tqdm import tqdm
import time


def log_mixing_weights(cfg, exp_folder, network, global_step, writer=None):
    mixing_weights = torch.nn.functional.softmax(network.mixing_weights, dim=0)
    num_layers = len(network.feature_dims)
    num_timesteps = len(network.save_timestep)
    save_timestep = network.save_timestep
    if cfg.diffusion_mode == "inversion":
        save_timestep = save_timestep[::-1]
    fig, ax = plt.subplots()
    ax.imshow(mixing_weights.view((num_timesteps, num_layers)).T.detach().cpu().numpy())
    ax.set_ylabel("Layer")  
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(range(1, num_layers+1))
    ax.set_xlabel("Timestep")
    ax.set_xticks(range(num_timesteps))
    ax.set_xticklabels(save_timestep)
    
    save_path = f"{exp_folder}/mixing_weighs/"
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(f"{save_path}/mixing_weights_step{global_step}.png")
    
    if cfg.report_to == "wandb":
        wandb.log({f"mixing_weights": plt}, step=global_step)
    if cfg.report_to == "tensorboard":
        fig_canvas = fig.canvas
        fig_canvas.draw()
        tb_img = torch.from_numpy(
            np.frombuffer(fig_canvas.buffer_rgba(), dtype='uint8')
            .reshape(fig_canvas.get_width_height()[::-1] + (4,))
        ).permute(2, 0, 1)
        tb_img = tb_img[:3] 
        writer.add_image("mixing_weights", tb_img, global_step)

    plt.close(fig)


def latents_to_images(latents, vae, device):
    latents = latents.to(device)
    latents = latents / 0.18215
    images = vae.decode(latents.to(vae.dtype)).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    return [Image.fromarray(image) for image in images]


class Evaluator:
    """Base class for evaluation across tasks."""
    def __init__(self, cfg, network, ldm_extractor, val_loader, vae, g, logger, val_type="val", writer=None):
        self.cfg = cfg
        self.network = network
        self.ldm_extractor = ldm_extractor
        self.val_loader = val_loader
        self.vae = vae
        self.g = g
        self.logger = logger
        self.val_type = val_type
        self.writer = writer
        self.device = cfg.device
        self.ignore_index = cfg.get("ignore_index", -100)
        self.num_classes = cfg.num_classes
        self.class_names = cfg.class_names if hasattr(cfg, "class_names") else list(range(self.num_classes))

    def log_metrics(self, metrics, epoch, global_step):
        """Log metrics to logger, WandB, and TensorBoard."""
        self.logger.info(f"Validation Metrics")
        for metric, value in metrics.items():
            if isinstance(value, list):
                for c, v in zip(self.class_names, value):
                    self.logger.info(f"{self.val_type}/{metric}_{c}: {v:.4f}")
                self.logger.info(f"{self.val_type}/m{metric}: {np.mean(value):.4f}")
            else:
                self.logger.info(f"{self.val_type}/{metric}: {value:.4f}")

        if self.cfg.report_to == "wandb":
            wandb_metrics = {f"{self.val_type}/{k}": v for k, v in metrics.items() if not isinstance(v, list)}
            for metric, values in metrics.items():
                if isinstance(values, list):
                    wandb_metrics.update({f"{self.val_type}/{metric}_{c}": v for c, v in zip(self.class_names, values)})
            wandb.log(wandb_metrics, step=global_step)

        if self.cfg.report_to == "tensorboard" and self.writer:
            for metric, value in metrics.items():
                if isinstance(value, list):
                    for c, v in zip(self.class_names, value):
                        self.writer.add_scalar(f"{self.val_type}/{metric}_{c}", v, global_step)
                    self.writer.add_scalar(f"{self.val_type}/m{metric}", np.mean(value), global_step)
                else:
                    self.writer.add_scalar(f"{self.val_type}/{metric}", value, global_step)

    def validate(self, epoch, global_step):
        raise NotImplementedError


class SingleLabelClsEvaluator(Evaluator):
    """Evaluator for single-label classification."""
    def validate(self, epoch, global_step):
        self.logger.info(f"Validation {self.val_type} set - Epoch {epoch} - Step {global_step}")
        start_time = time.time()
        self.network.eval()

        total_correct = 0
        total_samples = 0
        conf_matrix = np.zeros((self.num_classes, self.num_classes))
        true_positive = np.zeros(self.num_classes)
        false_positive = np.zeros(self.num_classes)
        false_negative = np.zeros(self.num_classes)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, total=len(self.val_loader), leave=False):
                images = batch['rgb'].to(self.device)
                labels = batch['label'].to(self.device)
                latents = self.vae.encode(images).latent_dist.sample(generator=self.g) * 0.18215
                latents = latents.to(self.device)
                
                with torch.inference_mode():
                    feats, _ = self.ldm_extractor.forward(latents)
                    logits, _ = self.network(feats, output_shape=None)
                    preds = torch.argmax(logits, dim=1)

                if self.ignore_index is not None:
                    mask = labels != self.ignore_index
                    if mask.sum().item() == 0:
                        continue
                    labels = labels[mask]
                    preds = preds[mask]

                batch_correct = (preds == labels).sum().item()
                total_correct += batch_correct
                total_samples += labels.size(0)

                batch_conf_matrix = confusion_matrix(
                    labels.cpu().numpy(), preds.cpu().numpy(), labels=range(self.num_classes)
                )
                conf_matrix += batch_conf_matrix
                true_positive += np.diag(batch_conf_matrix)
                false_positive += batch_conf_matrix.sum(axis=0) - np.diag(batch_conf_matrix)
                false_negative += batch_conf_matrix.sum(axis=1) - np.diag(batch_conf_matrix)

        accuracy = total_correct / total_samples
        precision = (true_positive / (true_positive + false_positive + 1e-10)).mean()
        recall = (true_positive / (true_positive + false_negative + 1e-10)).mean()
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
        self.logger.info("Confusion Matrix:")
        self.logger.info(conf_matrix)
        self.log_metrics(metrics, epoch, global_step)

        end_time = time.time()
        self.logger.info(f"Validation completed in {end_time - start_time:.2f} seconds")
        return accuracy

class MultiLabelClsEvaluator(Evaluator):
    """Evaluator for multi-label classification."""
    def validate(self, epoch, global_step):
        self.logger.info(f"Starting validation epoch {epoch} on {self.val_type} set")
        start_time = time.time()
        self.network.eval()

        all_targets = []
        all_preds = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, total=len(self.val_loader), leave=False):
                images = batch['rgb'].to(self.device)
                labels = batch['label'].to(self.device)
                latents = self.vae.encode(images).latent_dist.sample(generator=self.g) * 0.18215
                latents = latents.to(self.device)

                with torch.inference_mode():
                    feats, _ = self.ldm_extractor.forward(latents)
                    logits, _ = self.network(feats, output_shape=None)
                    preds = torch.sigmoid(logits) > 0.5
                all_targets.extend(labels.cpu().numpy())
                all_preds.extend(preds.type(labels.dtype).cpu().numpy())

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true=all_targets, y_pred=all_preds, average="micro", zero_division=0
        )
        metrics = {
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
        }
        self.log_metrics(metrics, epoch, global_step)

        end_time = time.time()
        self.logger.info(f"Validation epoch {epoch} completed in {end_time - start_time:.2f} seconds")
        return f1

class SegmentationEvaluator(Evaluator):
    """Evaluator for segmentation tasks."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_expert_colors = {k: np.random.randint(0, 255, (3,), dtype=np.uint8) for k in range(1000)}
        self.class_colors = (
            {i: (np.array(plt.get_cmap("tab20")(i)[:3]) * 255).astype(np.uint8) for i in range(self.num_classes)}
            if self.num_classes > 2 else {0: (0, 0, 0), 1: (255, 255, 255)}
        )

    def validate(self, epoch, global_step):
        self.logger.info(f"Starting validation epoch {epoch} on {self.val_type} set")
        start_time = time.time()
        self.network.eval()

        confusion_matrix = torch.zeros(self.num_classes, self.num_classes, device=self.device)

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(self.val_loader), total=len(self.val_loader), leave=False):
                images = batch['rgb'].to(self.device)
                labels = batch['label'].to(self.device)
                filenames = batch['filename']
                metadata = batch.get('metadata', {})

                latents = self.vae.encode(images).latent_dist.sample(generator=self.g) * 0.18215
                latents = latents.to(self.device)
                with torch.inference_mode():
                    feats, pred_x0s = self.ldm_extractor.forward(latents)
                    logits, _ = self.network(feats, output_shape=(self.cfg.original_img_size, self.cfg.original_img_size))
                    preds = torch.argmax(logits, dim=1)

                mask = (labels != self.ignore_index) if self.ignore_index is not None else torch.ones_like(labels, dtype=torch.bool)
                valid_preds = preds[mask]
                valid_labels = labels[mask]
                count = torch.bincount(
                    (valid_preds * self.num_classes + valid_labels).flatten(),
                    minlength=self.num_classes ** 2
                ) if valid_preds.numel() > 0 else torch.bincount(
                    (preds * self.num_classes + labels).flatten(),
                    minlength=self.num_classes ** 2
                )
                confusion_matrix += count.view(self.num_classes, self.num_classes)


        confusion_matrix = confusion_matrix.cpu()
        intersection = torch.diag(confusion_matrix)
        union = confusion_matrix.sum(dim=0) + confusion_matrix.sum(dim=1) - intersection
        iou = (intersection / (union + 1e-6)) * 100
        precision = (intersection / (confusion_matrix.sum(dim=0) + 1e-6)) * 100
        recall = (intersection / (confusion_matrix.sum(dim=1) + 1e-6)) * 100
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        metrics = {
            "IoU": iou.tolist(),
            "mIoU": iou.mean().item(),
            "F1": f1.tolist(),
            "mF1": f1.mean().item(),
            "Precision": precision.tolist(),
            "mPrecision": precision.mean().item(),
            "Recall": recall.tolist(),
            "mRecall": recall.mean().item(),
            "mAcc": (intersection.sum() / (confusion_matrix.sum() + 1e-6)).item() * 100,
        }
        self.log_metrics(metrics, epoch, global_step)

        end_time = time.time()
        self.logger.info(f"Validation epoch {epoch} completed in {end_time - start_time:.2f} seconds")
        return metrics["mIoU"]


def val_log(cfg, network, ldm_extractor, val_loader, vae, g, epoch, global_step, logger, val_type="val", writer=None):
    """Dispatch validation based on task type."""
    Evaluators = {
        "classification": SingleLabelClsEvaluator,
        "multi_label_classification": MultiLabelClsEvaluator,
        "segmentation": SegmentationEvaluator
    }
    task_key = cfg.get("task", None)
    if not task_key:
        raise ValueError(f"Task type {cfg.task} not supported")
    
    Evaluator = Evaluators[task_key](cfg, network, ldm_extractor, val_loader, vae, g, logger, val_type, writer)
    return Evaluator.validate(epoch, global_step)