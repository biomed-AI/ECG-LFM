import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import numpy as np
import torch.nn.functional as F
from fairseq_signals import logging, metrics, meters
from fairseq_signals.utils import utils
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.logging.meters import safe_round
from scipy.stats import pearsonr,spearmanr

@dataclass
class MSECriterionConfig(Dataclass):
    pass

@register_criterion("mse", dataclass=MSECriterionConfig)
class MSECriterion(BaseCriterion):
    def __init__(self, task):
        super().__init__(task)

    def compute_loss(
        self, logits, target, sample=None, net_output=None, model=None, reduce=True
    ):
        """
        Compute the loss given the logits and targets from the model
        """
        reduction = "none" if not reduce else "sum"
        loss = F.mse_loss(logits, target, reduction=reduction)

        return loss, [loss.detach().item()]

    def get_sample_size(self, sample, target):
        """
        Get the sample size, which is used as the denominator for the gradient
        """
        if 'sample_size' in sample:
            sample_size = sample['sample_size']
        else:
            sample_size = target.numel()
        return sample_size

    def get_logging_output(
        self, logging_output, logits=None, target=None, sample=None, net_output=None
    ):
        """
        Get the logging output to display while training
        """
        y_true = target.cpu().numpy()
        y_pred = logits.detach().cpu().numpy()
        logging_output['y_true'] = y_true
        logging_output['y_pred'] = y_pred
        if not self.training:
            logging_output['_y_true'] = y_true
            logging_output['_y_pred'] = y_pred
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
            f"{prefix}loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round = 3
        )
        metrics.log_scalar(f"{prefix}nsignals", nsignals)
        
        if "_y_true" in logging_outputs[0] and "_y_pred" in logging_outputs[0]:
            y_true = np.concatenate([log["_y_true"] for log in logging_outputs if "_y_true" in log])
            y_pred = np.concatenate([log["_y_pred"] for log in logging_outputs if "_y_pred" in log])
            y_class = [log["_y_class"] for log in logging_outputs if "_y_class" in log]
            metrics.log_custom(meters.AUCMeter, f"_{prefix}pcc", y_pred, y_true, y_class)
            if len(y_true) > 1:
                metrics.log_derived(
                    f"{prefix}pearsonr",
                    lambda meters: safe_round(
                        meters[f"_{prefix}pcc"].pearson, 3
                    )
                )
                metrics.log_derived(
                    f"{prefix}r2",
                    lambda meters: safe_round(
                        meters[f"_{prefix}pcc"].r2_score, 3
                    )
                )
        #metrics.log_scalar(f"{prefix}PCC", pearsonr(y_true.reshape(-1), y_pred.reshape(-1))[0])
        

    def logging_outputs_can_be_summed(self) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False