# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True,
                            tgt_wil=None, seg_indices=None, seg_weights=None):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    if tgt_wil.dim() == lprobs.dim() - 1:
        tgt_wil = tgt_wil.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if tgt_wil is not None and seg_indices is not None and seg_weights is not None:
        loss_weights = torch.ones_like(nll_loss)
        assert len(seg_indices) == len(seg_weights)
        for seg_i, seg_w in zip(seg_indices, seg_weights):
            seg_mask = tgt_wil.eq(seg_i)
            loss_weights.masked_fill_(seg_mask, seg_w)
        nll_loss *= loss_weights
        smooth_loss *= loss_weights
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy_split')
class LabelSmoothedCrossEntropyCriterionSplit(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing,
                 ls_segment_indices, ls_segment_decay):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        ls_seg_indices = ls_segment_indices.split(',')
        self.ls_seg_indices = [int(x) for x in ls_seg_indices]
        ls_seg_decay = ls_segment_decay.split(',')
        self.ls_seg_decay = [float(x) for x in ls_seg_decay]
        assert len(self.ls_seg_indices) == len(self.ls_seg_decay)
        self.ls_seg_weights = [1.0 for _ in self.ls_seg_indices]

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--ls-segment-indices', type=str, default='0,1',
                            help='indices of the segments')
        parser.add_argument('--ls-segment-decay', type=str, default='1,0',
                            help='weights of the segments')
        # fmt: on

    def forward(self, model, sample, reduce=True,
                lambda_decay=None, lambda_pseudo_decay=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if lambda_decay is not None and lambda_pseudo_decay is None:
            self.ls_seg_weights = [lambda_decay if di == 1 else 2.0 - lambda_decay for di in self.ls_seg_decay]
        if lambda_decay is not None and lambda_pseudo_decay is not None:
            ls_seg_weights = []
            for idx, dflag in zip(self.ls_seg_indices, self.ls_seg_decay):
                if idx < 50:
                    temp_weight = lambda_decay if dflag == 1 else 2.0 - lambda_decay
                else:
                    temp_weight = lambda_pseudo_decay if dflag == 1 else 2.0 - lambda_pseudo_decay
                ls_seg_weights.append(temp_weight)
            self.ls_seg_weights = ls_seg_weights

        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        tgt_wil = sample['tgt_wil'].view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            tgt_wil=tgt_wil,seg_indices=self.ls_seg_indices,seg_weights=self.ls_seg_weights
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
