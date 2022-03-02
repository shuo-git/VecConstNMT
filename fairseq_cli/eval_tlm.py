#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""

import logging
import math
import os
import sys

import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import LMContextWindowDataset
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_scorer import SequenceScorer
from fairseq import distributed_utils


logging.basicConfig(
    format='%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.eval_tlm')


def main(parsed_args, **unused_kwargs):
    assert parsed_args.path is not None, '--path required for evaluation!'

    if torch.cuda.is_available() and not parsed_args.cpu:
        torch.cuda.set_device(parsed_args.device_id)

    utils.import_user_module(parsed_args)

    if parsed_args.max_tokens is None and parsed_args.max_sentences is None:
        parsed_args.max_tokens = 12000

    logger.info(parsed_args)

    use_cuda = torch.cuda.is_available() and not parsed_args.cpu

    task = tasks.setup_task(parsed_args)
    task.load_dataset(parsed_args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    logger.info('loading model(s) from {}'.format(parsed_args.path))
    models, _args = checkpoint_utils.load_model_ensemble(
        parsed_args.path.split(os.pathsep),
        arg_overrides=eval(parsed_args.model_overrides),
        task=task,
        suffix=getattr(parsed_args, "checkpoint_suffix", ""),
    )

    # Optimize ensemble for generation
    for model in models:
        # model.prepare_for_inference_(parsed_args)
        if parsed_args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(parsed_args.gen_subset),
        max_tokens=parsed_args.max_tokens,
        max_sentences=parsed_args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=parsed_args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=parsed_args.required_batch_size_multiple,
        num_shards=parsed_args.num_shards,
        shard_id=parsed_args.shard_id,
        num_workers=parsed_args.num_workers,
        data_buffer_size=parsed_args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=parsed_args.log_format,
        log_interval=parsed_args.log_interval,
        default_log_format=('tqdm' if not parsed_args.no_progress_bar else 'none'),
    )

    # Initialize scorer
    gen_timer = StopwatchMeter()
    scorer = SequenceScorer(tgt_dict, parsed_args.softmax_batch)

    score_sum = 0.
    count = 0

    wps_meter = TimeMeter()

    for sample in progress:
        if 'net_input' not in sample:
            continue

        sample = utils.move_to_cuda(sample) if use_cuda else sample

        gen_timer.start()
        hypos = scorer.generate(models, sample)
        gen_timer.stop(sample['ntokens'])

        for i, hypos_i in enumerate(hypos):
            hypo = hypos_i[0]
            sample_id = sample['id'][i]

            tokens = hypo['tokens']
            tgt_len = tokens.numel()
            pos_scores = hypo['positional_scores'].float()

            inf_scores = pos_scores.eq(float('inf')) | pos_scores.eq(float('-inf'))
            if inf_scores.any():
                logger.info(
                    'skipping tokens with inf scores:',
                    tgt_dict.string(tokens[inf_scores.nonzero()])
                )
                pos_scores = pos_scores[(~inf_scores).nonzero()]
            score_sum += pos_scores.sum().cpu()
            count += pos_scores.numel()

            if parsed_args.output_word_probs:
                w = ''
                word_prob = []
                for w_i in range(tgt_len):
                    w_ind = tokens[w_i].item()
                    w += tgt_dict[w_ind]
                    word_prob.append((w, pos_scores[w_i].item()))
                    w = ''
                logger.info(
                    "T-" + str(int(sample_id)) + "\t"
                    + (' '.join('{}'.format(x[0]) for x in word_prob[:-1]))
                )
                logger.info(
                    "P-" + str(int(sample_id)) + "\t"
                    + (' '.join('{:.4f}'.format(x[1]) for x in word_prob))
                )

        wps_meter.update(sample['ntokens'])
        progress.log({'wps': round(wps_meter.avg)})

    avg_nll_loss = -score_sum / count / math.log(2)  # convert to base 2
    logger.info('Evaluated {} tokens in {:.1f}s ({:.2f} tokens/s)'.format(
        gen_timer.n, gen_timer.sum, 1. / gen_timer.avg
    ))
    logger.info('Loss (base 2): {:.4f}, Perplexity: {:.2f}'.format(
        avg_nll_loss, 2**avg_nll_loss
    ))


def cli_main():
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == '__main__':
    cli_main()
