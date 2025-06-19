import math
import logging

from dataclasses import dataclass, field
from typing import List, Optional
from fairseq_signals.logging.metrics import aggregate

import torch
import torch.nn.functional as F
from fairseq_signals import metrics
from fairseq_signals.utils import utils
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.logging.meters import safe_round

logger = logging.getLogger(__name__)
@dataclass
class ArcFaceCriterionConfig(Dataclass):
    scale: float = field(
        default=32,
        metadata={
            "help": "scaling factor in marginal cosine similarity"
        }
    )
    margin: float = field(
        default=0.5,
        metadata={
            "help": "angular margin penalty between two vectors"
        }
    )

@register_criterion("arcface", dataclass=ArcFaceCriterionConfig)
class ArcFaceCriterion(BaseCriterion):
    def __init__(self, task, scale=32, margin=0.5, log_keys=None):
        super().__init__(task)
        self.scale = scale
        self.margin = margin
        self.log_keys = [] if log_keys is None else log_keys

    def compute_loss(
        self, logits, target, sample=None, net_output=None, model=None, reduce=True
    ):
        """
        Compute the loss given the final logits and targets directly fed to the loss function
        """
        cos_m = math.cos(self.margin)
        sin_m = math.sin(self.margin)
        mm = sin_m * self.margin
        threshold = math.cos(math.pi - self.margin)

        cos_theta = logits
        sin_theta_2 = 1 - torch.pow(cos_theta, 2)
        sin_theta = torch.sqrt(sin_theta_2)

        cos_theta_m = (cos_theta * cos_m - sin_theta * sin_m)

        cond_v = cos_theta - threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - mm)

        cos_theta_m[cond_mask] = keep_val[cond_mask]
        logits = cos_theta * 1.0
        idx = torch.arange(0, cos_theta.size(0))

        logits[idx, target] = cos_theta_m[idx, target]
        logits *= self.scale

        reduction = "none" if not reduce else "sum"
        loss = F.cross_entropy(
            input=logits,
            target=target,
            reduction=reduction
        )

        return loss, [loss.detach().item()]

    def get_sample_size(self, sample, target):
        """
        Get the sample size, which is used as the denominator for the gradient
        """
        if "sample_size" in sample:
            sample_size = sample["sample_size"]
        else:
            sample_size = target.numel()
        return sample_size

    def get_logging_output(self, logging_output, logits, target, sample=None, net_output=None):
        """
        Get the logging output to display while training
        """
        with torch.no_grad():
            if logits.numel() == 0:
                corr = 0
                count = 0
            else:
                assert logits.dim() > 1, logits.shape
                
                output = (logits.data * 1.0).argmax(-1)
                count = float(output.numel())
                corr = (output == target).sum().item()
            
            logging_output["correct"] = corr
            logging_output["count"] = count
        return logging_output
    
    @staticmethod
    def reduce_metrics(logging_outputs, prefix: str = None) -> None:
        """Aggregate logging outputs from data parallel training."""
        if prefix is None:
            prefix = ""
        elif prefix is not None and not prefix.endswith("_"):
            prefix = prefix + "_"

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nsignals = utils.item(
            sum(log.get("nsignals", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            f"{prefix}loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round=3
        )

        if nsignals > 0:
            metrics.log_scalar(f"{prefix}nsignals", nsignals)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar(f"_{prefix}correct", correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar(f"_{prefix}total", total)

        if total > 0:
            metrics.log_derived(
                f"{prefix}accuracy",
                lambda meters: safe_round(
                    meters[f"_{prefix}correct"].sum / meters[f"_{prefix}total"].sum, 5
                )
                if meters[f"_{prefix}total"].sum > 0
                else float("nan")
            )

        builtin_keys = {
            "loss",
            "ntokens",
            "nsignals",
            "sample_size",
            "correct",
            "count"
        }

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k,0) for log in logging_outputs)
                if k.startswith("loss"):
                    metrics.log_scalar(
                        prefix + k, val / (sample_size or 1) / math.log(2), sample_size, round=3
                    )
                else:
                    metrics.log_scalar(prefix + k, val / len(logging_outputs), round=3)

    def logging_outputs_can_be_summed(self) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False