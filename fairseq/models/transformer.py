# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    Target_Plug_In_Layer_Type1,
    Target_Plug_In_Layer_Type2,
    Target_Plug_In_Layer_Type3,
    Softmax_Plug_In_Gate,
)
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
import torch.nn.functional as F

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("transformer")
class TransformerModel(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_subword(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'subword_nmt',
            }

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        return {
            'transformer.wmt14.en-fr': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            'transformer.wmt19.en-de': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.en-ru': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz'),
            'transformer.wmt19.de-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.ru-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz'),
            'transformer.wmt19.en-de.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.en-ru.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz'),
            'transformer.wmt19.de-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.ru-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz'),
        }
        # fmt: on

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        parser.add_argument('--word-dropout', action='store_true', default=False,
                            help='if True, use word dropout on embedding layer of decoder')
        parser.add_argument('--word-dropout-prob', type=float, metavar='D', default=0,
                            help='word dropout probability')
        # attn plug-in parameters
        parser.add_argument('--target-kv-table', action='store_true', default=False)
        parser.add_argument('--encoder-out-key', action='store_true', default=False)
        parser.add_argument('--plug-in-type', default='type1',
                            help='type1 or type2 or type3 or ...')
        parser.add_argument('--plug-in-forward', default='bottom',
                            help='bottom or pipe')
        parser.add_argument('--plug-in-project', type=str, default='both',
                            help='both or key or value or none')
        parser.add_argument('--plug-in-component', type=str, default='encdec',
                            help='encdec or enc or dec or none')
        parser.add_argument('--plug-in-v-project', action='store_true', default=False)
        parser.add_argument('--plug-in-k-project', action='store_true', default=False)
        parser.add_argument('--plug-in-dec-self-attn', action='store_true', default=False)
        parser.add_argument('--aggregator-v-project', action='store_true', default=False)
        parser.add_argument('--plug-in-mid-dim', type=int, metavar='D', default=512)
        parser.add_argument('--no-plug-in-pointer', action='store_true', default=False)
        parser.add_argument('--no-plug-in-pointer-gate', action='store_true', default=False)
        parser.add_argument('--kv-attention-dropout', type=float, metavar='D', default=0.0,
                            help='dropout probability for kv attention weights')
        parser.add_argument('--kv-projection-dropout', type=float, metavar='D', default=0.0,
                            help='dropout probability for kv projections')
        parser.add_argument('--plug-in-enc-layers', type=str, default='0,1,2,3,4,5')
        parser.add_argument('--plug-in-dec-layers', type=str, default='0,1,2,3,4,5')
        # baseline pointer network parameters
        parser.add_argument('--source-pointer', action='store_true', default=False)
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        if args.target_kv_table and args.plug_in_component != 'none':
            kv_aggregator = cls.build_kv_aggregator(
                args, args.encoder_embed_dim
            )
        else:
            kv_aggregator = None

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens, kv_aggregator)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens, kv_aggregator)
        return cls(args, encoder, decoder)

    @classmethod
    def build_kv_aggregator(cls, args, embed_dim):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.kv_attention_dropout,
            encoder_decoder_attention=True,
            aggregator_v_project=args.aggregator_v_project,
        )

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens, kv_aggregator):
        return TransformerEncoder(args, src_dict, embed_tokens, kv_aggregator)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, kv_aggregator):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            kv_aggregator,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        **kwargs,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens, **kwargs
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            **kwargs,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens, kv_aggregator):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self.args = args
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.num_predix_tokens = 1000
        self.normal_token_threshold = self.embed_tokens.weight.shape[0] - self.num_predix_tokens - 1
        self.embed_prefix = self.build_prefix(args, self.num_predix_tokens, embed_dim)

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.kv_aggregator = kv_aggregator
        # Build the Plug-In network
        self.plug_in_layers = [int(pl) for pl in self.args.plug_in_enc_layers.split(',')]
        self.plug_in_layer_map = {}
        for idx, pl in enumerate(self.plug_in_layers):
            self.plug_in_layer_map[pl] = idx
        if self.args.target_kv_table and 'enc' in self.args.plug_in_component:
            self.plug_ins = nn.ModuleList([])
            self.plug_ins.extend(
                [
                    build_plug_in_layer(args, embed_dim, args.encoder_attention_heads)
                    for _ in range(len(self.plug_in_layers))
                ]
            )
        else:
            self.plug_ins = None

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def build_prefix(self, args, num_embed, embed_dim):
        prefix_emb = nn.Embedding(num_embed + 1, embed_dim, padding_idx=0)
        nn.init.normal_(prefix_emb.weight, mean=0, std=embed_dim ** -0.5)
        nn.init.constant_(prefix_emb.weight[0], 0)
        return prefix_emb

    def forward_prefix_embedding(self, src_tokens):
        prefix_tokens = src_tokens * (src_tokens > self.normal_token_threshold) \
                        + (src_tokens <= self.normal_token_threshold) * self.normal_token_threshold \
                        - self.normal_token_threshold
        return self.embed_prefix(prefix_tokens)

    def forward_normal_embedding(self, src_tokens):
        return self.embed_tokens(src_tokens)

    def build_encoder_layer(self, args):
        return TransformerEncoderLayer(args)

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        mask_prefix = torch.unsqueeze(src_tokens > self.normal_token_threshold, -1)
        temp_prefix_embed = self.forward_prefix_embedding(src_tokens)
        mask_normal = torch.unsqueeze(src_tokens <= self.normal_token_threshold, -1)
        temp_normal_embed = self.forward_normal_embedding(src_tokens)
        temp_total_embed = mask_prefix * temp_prefix_embed + mask_normal * temp_normal_embed
        x = embed = self.embed_scale * temp_total_embed
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(self, src_tokens, src_lengths, return_all_hiddens: bool = False, **kwargs):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(src_tokens)

        target_key = kwargs.get('target_key', None)
        target_value = kwargs.get('target_value', None)
        bsz = x.shape[0]
        embed_dim = x.shape[-1]
        if self.args.target_kv_table and target_key is not None and target_value is not None and self.args.plug_in_component != 'none':
            target_key = target_key.view(bsz * 3, -1)
            target_value = target_value.view(bsz * 3, -1)
            tgt_k, _ = self.forward_embedding(target_key)
            tgt_k = tgt_k.transpose(0, 1) # T(k) x 3B x C
            tgt_v, _ = self.forward_embedding(target_value)
            tgt_v = tgt_v.transpose(0, 1) # T(v) x 3B x C
            tgt_v_padding_mask = target_value.eq(self.padding_idx) # 3B x T(v)
            tgt_v, _ = self.kv_aggregator(
                query=tgt_k,
                key=tgt_v,
                value=tgt_v,
                key_padding_mask=tgt_v_padding_mask,
            ) # T(k) x 3B x V
            tgt_k = tgt_k.transpose(0, 1).view(bsz, -1, embed_dim).transpose(0, 1) # 3T(k) x B x C
            tgt_v = tgt_v.transpose(0, 1).contiguous().view(bsz, -1, embed_dim).transpose(0, 1) # 3T(k) x B x C
            tgt_k_padding_mask = target_key.view(bsz, -1).eq(self.padding_idx) # B x 3T(k)

            target_value = target_value.view(bsz, -1) # B x 3T(v)
            tgt_v_embed, _ = self.forward_embedding(target_value)
            tgt_v_embed = tgt_v_embed.transpose(0, 1) # 3T(v) x B x C
            tgt_v_padding_mask = target_value.view(bsz, -1) # B x 3T(v)
            attend_kv_table = True
        else:
            tgt_k = None
            tgt_v = None
            tgt_k_padding_mask = None
            tgt_v_embed = None
            tgt_v_padding_mask = None
            attend_kv_table = False

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        if attend_kv_table:
            temp_tgt_k = tgt_k
            temp_tgt_v = tgt_v
        for idx, layer in enumerate(self.layers):
            if attend_kv_table and 'enc' in self.args.plug_in_component and idx in self.plug_in_layers:
                if self.args.plug_in_forward == 'bottom':
                    temp_tgt_k, temp_tgt_v = self.plug_ins[self.plug_in_layer_map[idx]](tgt_k, tgt_v)
                else:
                    temp_tgt_k, temp_tgt_v = self.plug_ins[self.plug_in_layer_map[idx]](temp_tgt_k, temp_tgt_v)
            else:
                temp_tgt_k = temp_tgt_v = None
            x = layer(
                    x,
                    encoder_padding_mask,
                    past_key=temp_tgt_k,
                    past_value=temp_tgt_v,
                    past_key_padding_mask=tgt_k_padding_mask,
                    past_kv_forward=self.args.plug_in_project,
                )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            tgt_k=tgt_k,  # 3T(k) x B x C
            tgt_v=tgt_v,  # 3T(k) x B x C
            tgt_k_padding_mask=tgt_k_padding_mask,  # B x 3T(k)
            tgt_v_embed=tgt_v_embed,  # 3T(v) x B x C
            tgt_v_padding_mask=tgt_v_padding_mask,  # B x 3T(v)
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        tgt_k = encoder_out.tgt_k
        if tgt_k is not None:
            tgt_k = tgt_k.index_select(1, new_order)
        tgt_v = encoder_out.tgt_v
        if tgt_v is not None:
            tgt_v = tgt_v.index_select(1, new_order)
        tgt_k_padding_mask = encoder_out.tgt_k_padding_mask
        if tgt_k_padding_mask is not None:
            tgt_k_padding_mask = tgt_k_padding_mask.index_select(0, new_order)

        tgt_v_embed = encoder_out.tgt_v_embed
        if tgt_v_embed is not None:
            tgt_v_embed = tgt_v_embed.index_select(1, new_order)
        tgt_v_padding_mask = encoder_out.tgt_v_padding_mask
        if tgt_v_padding_mask is not None:
            tgt_v_padding_mask = tgt_v_padding_mask.index_select(0, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
            tgt_k=tgt_k,  # 3T(k) x B x C
            tgt_v=tgt_v,  # 3T(k) x B x C
            tgt_k_padding_mask=tgt_k_padding_mask,  # B x 3T(k)
            tgt_v_embed=tgt_v_embed,  # 3T(v) x B x C
            tgt_v_padding_mask=tgt_v_padding_mask,  # B x 3T(v)
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, kv_aggregator, no_encoder_attn=False):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        assert embed_dim == input_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        # self.project_in_dim = (
        #     Linear(input_embed_dim, embed_dim, bias=False)
        #     if embed_dim != input_embed_dim
        #     else None
        # )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if self.args.language_embedding_num > 0:
            self.embed_languages = Embedding(self.args.language_embedding_num + 1, embed_dim,
                                             self.args.language_embedding_num)
        else:
            self.embed_languages = None

        # Build Plug-In network
        self.plug_in_layers = [int(pl) for pl in self.args.plug_in_dec_layers.split(',')]
        self.plug_in_layer_map = {}
        for idx, pl in enumerate(self.plug_in_layers):
            self.plug_in_layer_map[pl] = idx
        if self.args.target_kv_table:
            self.plug_ins = nn.ModuleList([])
            if 'dec' in self.args.plug_in_component:
                self.plug_ins.extend(
                    [
                        build_plug_in_layer(args, embed_dim, args.decoder_attention_heads)
                        for _ in range(len(self.plug_in_layers))
                    ]
                )
            if not self.args.no_plug_in_pointer and not self.args.no_plug_in_pointer_gate:
                self.plug_ins.append(self.build_softmax_plug_in(args, embed_dim, args.decoder_attention_heads))
            # if self.args.plug_in_dec_self_attn:
            #     self.self_plug_ins = nn.ModuleList([])
            #     self.self_plug_ins.extend(
            #         [
            #             build_plug_in_layer(args, embed_dim, args.decoder_attention_heads)
            #             for _ in range(len(self.plug_in_layers))
            #         ]
            #     )
            # else:
            #     self.self_plug_ins = None
        else:
            self.plug_ins = None
            # self.self_plug_ins = None

        # Build Source Pointer Network
        if self.args.source_pointer:
            p_gen_input_size = input_embed_dim + self.output_embed_dim
            self.project_p_gens = nn.Linear(p_gen_input_size, 1)
            nn.init.zeros_(self.project_p_gens.bias)

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = None
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )

    def build_softmax_plug_in(self, args, my_dim, head_num):
        return Softmax_Plug_In_Gate(args, my_dim, head_num)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return TransformerDecoderLayer(args, no_encoder_attn)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        **kwargs,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            **kwargs,
        )
        
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        **kwargs,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            **kwargs,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. Aa copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        **kwargs,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )
        # embed language indices
        prev_output_wil = None
        if self.embed_languages is not None and prev_output_wil is not None:
            lang_indices = self.embed_languages(prev_output_wil)
        else:
            lang_indices = None

        if incremental_state is not None:
            current_time_step = prev_output_tokens.shape[1]
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
            if lang_indices is not None:
                lang_indices = lang_indices[:, -1:]

        if kwargs.get('target_key', None) is not None and kwargs.get('target_value', None) is not None and self.args.target_kv_table:
            attend_kv_table = True
        else:
            attend_kv_table = False

        # embed tokens and positions
        bsz = prev_output_tokens.shape[0]
        seq_len = prev_output_tokens.shape[1]
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if attend_kv_table:
            tgt_k = encoder_out.tgt_k
            tgt_v = encoder_out.tgt_v
            tgt_k_padding_mask = encoder_out.tgt_k_padding_mask
            tgt_v_embed = encoder_out.tgt_v_embed
            tgt_v_padding_mask = encoder_out.tgt_v_padding_mask
            tgt_k_toks = kwargs.get('target_key').view(bsz, -1)
            tgt_v_toks = kwargs.get('target_value').view(bsz, -1)
        else:
            tgt_k = tgt_v = tgt_v_embed = None
            tgt_k_padding_mask = tgt_v_padding_mask = None
            tgt_k_toks = tgt_v_toks = None

        # if self.quant_noise is not None:
        #     x = self.quant_noise(x)

        # if self.project_in_dim is not None:
        #     x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if lang_indices is not None and self.args.use_learned_language_embedding:
            x += lang_indices

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
            if attend_kv_table:
                tgt_k = self.layernorm_embedding(tgt_k)
                tgt_v = self.layernorm_embedding(tgt_v)
                tgt_v_embed = self.layernorm_embedding(tgt_v_embed)

        if self.args.word_dropout:
            if self.training:
                x = F.dropout2d(x, p=self.args.word_dropout_prob, training=True, inplace=False)
        else:
            x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        if attend_kv_table:
            temp_tgt_k = tgt_k
            temp_tgt_v = tgt_v
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if attend_kv_table and 'dec' in self.args.plug_in_component and idx in self.plug_in_layers:
                if self.args.plug_in_forward == 'bottom':
                    temp_tgt_k, temp_tgt_v = self.plug_ins[self.plug_in_layer_map[idx]](tgt_k, tgt_v)
                    # temp_self_k, temp_self_v = self.self_plug_ins[self.plug_in_layer_map[idx]](tgt_v_embed, tgt_v_embed)
                else:
                    temp_tgt_k, temp_tgt_v = self.plug_ins[self.plug_in_layer_map[idx]](temp_tgt_k, temp_tgt_v)
            else:
                temp_tgt_k = temp_tgt_v = None
                temp_self_k = temp_self_v = None

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                past_key=temp_tgt_k,
                past_value=temp_tgt_v,
                past_key_padding_mask=tgt_k_padding_mask,
                past_kv_forward=self.args.plug_in_project,
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0) # B x T(tgt) x T(src)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        if attend_kv_table and self.args.plug_in_component != 'none':
            tgt_k = tgt_k.transpose(0, 1)
            tgt_v = tgt_v.transpose(0, 1)

        def _cosine_similarity(am, bm, eps):
            a_n, b_n = am.norm(dim=-1, keepdim=True), bm.norm(dim=-1, keepdim=True)
            a_norm = am / torch.max(a_n, eps * torch.ones_like(a_n))
            b_norm = bm / torch.max(b_n, eps * torch.ones_like(b_n))
            sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
            return sim_mt

        epsilon = 1e-12
        logits = self.output_layer(x)
        model_prob = utils.softmax(logits, dim=-1) + epsilon # B x T x V, fp32
        rank_reg = 0.

        if self.args.source_pointer:
            prev_output_embed = self.embed_tokens(prev_output_tokens)
            prev_output_embed *= self.embed_scale
            predictors = torch.cat((prev_output_embed, x), 2)
            p_gens = self.project_p_gens(predictors)
            p_gens = torch.sigmoid(p_gens.float())
            src_tokens = encoder_out.src_tokens
            src_length = src_tokens.shape[1]
            gen_dists = torch.mul(model_prob, p_gens)
            assert attn is not None
            attn = torch.mul(attn.float(), 1 - p_gens)
            index = src_tokens.unsqueeze(1)
            index = index.expand(bsz, seq_len, src_length)
            attn_dists_size = (bsz, seq_len, model_prob.shape[-1])
            attn_dists = attn.new_zeros(attn_dists_size)
            attn_dists.scatter_add_(2, index, attn.float())
            model_prob = gen_dists + attn_dists

        if attend_kv_table and not self.args.no_plug_in_pointer:
            # Decoding Rule-1 by Shuo
            if incremental_state is not None:
                saved_state = self.get_incremental_state(incremental_state, "plug_in_state")
                if saved_state is not None:
                    tgt_v_toks = saved_state['target_value']
                else:
                    saved_state = {'target_value': tgt_v_toks}
                selected_tok = prev_output_tokens # B x 1
                selected_mask = tgt_v_toks.eq(selected_tok)
                tgt_v_toks = tgt_v_toks * (~selected_mask) + self.padding_idx * selected_mask # B x T(v)
                saved_state['target_value'] = tgt_v_toks
                self.set_incremental_state(incremental_state, 'plug_in_state', saved_state)

            tgt_v_padding_mask = tgt_v_toks.eq(self.padding_idx) # B x T(v)
            last_tgt_v = self.embed_scale * self.embed_tokens(tgt_v_toks) * (~tgt_v_padding_mask).unsqueeze(-1) # B x T(v) x C
            cos_sim = _cosine_similarity(x.float(), last_tgt_v.float(), epsilon) # B x T x T(v), need to be regularized
            cos_sim.masked_fill_(tgt_v_padding_mask.unsqueeze(1), 0.)
            if self_attn_padding_mask is not None:
                cos_sim.masked_fill_(self_attn_padding_mask.unsqueeze(-1), 0.)
            
            # cos sim rank regularization
            # norm_cos_sim = torch.nn.functional.normalize(cos_sim, dim=-1, eps=epsilon)
            # rank_reg = torch.bmm(norm_cos_sim, norm_cos_sim.transpose(1, 2))
            # rank_reg.masked_fill_(torch.eye(seq_len).to(rank_reg).to(bool).unsqueeze(0), 0.) # B x T x T
            # rank_reg = rank_reg.sum() # lower is better

            plug_in_sim = cos_sim.max(dim=-1, keepdim=True).values # B x T x 1
            # plug_in_sim = torch.max(torch.ones_like(plug_in_sim) * epsilon, plug_in_sim) # no negative plug-in-sim
            plug_in_v_idx = cos_sim.argmax(dim=-1) # B x T
            plug_in_v = tgt_v_toks.gather(index=plug_in_v_idx, dim=-1).unsqueeze(-1) # B x T x 1
            plug_in_prob = torch.zeros_like(model_prob).scatter(dim=-1, index=plug_in_v, src=plug_in_sim) # B x T x V
            
            if not self.args.no_plug_in_pointer_gate:
                plug_in_gate = self.plug_ins[-1](x.transpose(0, 1), last_tgt_v.transpose(0, 1), tgt_v_padding_mask).transpose(0, 1) # B x T x 1
                plug_in_gate = plug_in_gate.float()
                # Decoding Rule-2 by Shuo
                # if incremental_state is not None:
                #     plug_in_gate *= (1.0 * math.exp(0.02 * current_time_step))

                model_prob += plug_in_prob * plug_in_gate
            
            model_prob = torch.min(torch.ones_like(model_prob), model_prob) # < 1
            model_prob = torch.max(torch.ones_like(model_prob) * epsilon, model_prob) # > 0

            # Decoding Rule-3 by Shuo
            # if incremental_state is not None:
            #     assert saved_state is not None
            #     may_end_mask = torch.all(tgt_v_toks.eq(self.padding_idx), dim=-1, keepdim=True) # B x 1
            #     may_end_idx = (self.padding_idx * may_end_mask + self.dictionary.eos() * (~may_end_mask)).unsqueeze(-1).long() # B x 1 x 1
            #     may_end_src = torch.zeros_like(may_end_idx).float() + epsilon
            #     model_prob = model_prob.scatter(dim=-1, index=may_end_idx, src=may_end_src)

        return logits, {"attn": [attn], "inner_states": inner_states, "model_prob": model_prob, "rank_reg": rank_reg}

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        input_buffer = self.get_incremental_state(incremental_state, 'plug_in_state')
        if input_buffer is not None:
            for k in input_buffer:
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self.set_incremental_state(incremental_state, 'plug_in_state', input_buffer)
        return incremental_state

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def build_plug_in_layer(args, my_dim, head_num):
    if args.plug_in_type == 'type1':
        return Target_Plug_In_Layer_Type1(args, my_dim, head_num)
    elif args.plug_in_type == 'type2':
        return Target_Plug_In_Layer_Type2(args, my_dim, head_num)
    elif args.plug_in_type == 'type3':
        return Target_Plug_In_Layer_Type3(args, my_dim, head_num)


@register_model_architecture("transformer", "transformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)


@register_model_architecture("transformer", "transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("transformer", "transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, "dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture("transformer", "transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture("transformer", "transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)
