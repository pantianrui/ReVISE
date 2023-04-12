# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from dataclasses import dataclass, field
import torch
from scipy.io import wavfile
import os
from torch.nn.utils.rnn import pad_sequence

@dataclass
class MSELossCriterionConfig(FairseqDataclass):
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask "
            "(normalized by length)"
        },
    )
    sentence_avg: bool = II("optimization.sentence_avg")

@register_criterion("mse_loss", dataclass=MSELossCriterionConfig)
class MSELossCriterion(FairseqCriterion):
    def __init__(self, task,sentence_avg):
        super().__init__(task)
        self.mse = torch.nn.MSELoss(reduction='mean')
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outp
        
        uts to display while training
        """
        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        """
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        """
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)
        
    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_mse_output(net_output) 
        #print("lprobs.shape\n",lprobs.shape) #([24, 22720])
        #print("GGGGGG\n",sample['utt_id']) #dict_keys(['id', 'net_input', 'utt_id', 'target_lengths', 'ntokens', 'target'])
        target = [torch.from_numpy(wavfile.read(os.path.join('/home/pantianrui/data/lrs3/lrs3/audio/',uid+'.wav'))[1]) for uid in sample['utt_id']]
        target = pad_sequence(target, batch_first=True, padding_value=0)
        target = target.to(lprobs.device)
        target = target.type_as(lprobs)
        #print("target.shape\n",target.shape) #([24, 22528])
        target_dim = target.shape[1]
        lprobs_dim = lprobs.shape[1]
        if lprobs_dim > target_dim:
           lprobs = lprobs[:,:target_dim]
        elif lprobs_dim < target_dim:
           dif = target_dim - lprobs_dim
           lprobs = F.pad(lprobs, (0,dif, 0, 0), mode='constant', value=0)
        loss = self.mse(
            lprobs,
            target,
        )
        return loss, loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
