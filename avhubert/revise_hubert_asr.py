# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys,logging
import contextlib
import tempfile
from argparse import Namespace
from typing import Any, Optional,List

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, FairseqEncoderDecoderModel, register_model
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from fairseq.tasks import FairseqTask
from omegaconf import II, MISSING
from python_speech_features import logfbank
from .text_to_speech.vocoder import HiFiGANVocoder
import torchaudio
import torchlibrosa as tl
import pdb

DBG=True if len(sys.argv) == 1 else False

if DBG:
    from hubert import AVHubertModel
    from decoder import TransformerDecoder
else:
    from .hubert import AVHubertModel
    from .decoder import TransformerDecoder

logger = logging.getLogger(__name__)

@dataclass
class AVHubertAsrConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to hubert model"}
    )
    no_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "if true, does not load pretrained weights"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout after transformer and before final projection"
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability inside hubert model"},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights "
            "inside hubert model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
            "inside hubert model"
        },
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask "
            "(normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )
    feature_grad_mult: float = field(
        default=0.0,
        metadata={"help": "reset feature grad mult in hubert to this"},
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a layer in hubert"},
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")

    # this holds the loaded hubert args
    w2v_args: Any = None

@dataclass
class AVHubertCycleConfig(AVHubertAsrConfig):
    ###############for vocoder########################
    vocoder_path: str = field(
        default='/home/pantianrui/data/av_hubert/avhubert/checkpoint/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur/model.pt',metadata={"help":"path to vocoder pretrained model"}
    )
    resblock_kernel_sizes: list = field(
        default_factory=lambda:[3,7,11],metadata={"help":"resblock_kernel_sizes"}
    )
    upsample_rates: list = field(
        default_factory=lambda:[5,4,4,2,2],metadata={"help":"upsample_rates"}
    )
    model_in_dim: int = field(
        default=768,metadata={"help":"model_in_dim"}
    )
    upsample_initial_channel: int = field(
        default=512,metadata={"help":"upsample_initial_channel"}
    )
    upsample_kernel_sizes: list = field(
        default_factory=lambda:[11,8,8,4,4],metadata={"help":"upsample_kernel_sizes"}
    )
    resblock_dilation_sizes: list = field(
        default_factory=lambda:[[1,3,5], [1,3,5], [1,3,5]],metadata={"help":"resblock_dilation_sizes"}
    )
    #####################################################

class HubertEncoderWrapper(FairseqEncoder):
    def __init__(self, w2v_model):
        super().__init__(None)
        self.w2v_model = w2v_model

    def forward(self, source, padding_mask, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
        }

        #print('source[audio].shape\n',source['audio'].shape) #(7,104,138)
        x, padding_mask = self.w2v_model.extract_finetune(**w2v_args)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def get_normalized_probs(self,net_output, log_probs,sample):
        return net_output[0]

@register_model("revise_avhubert", dataclass=AVHubertCycleConfig)
class AVHubertCycle(BaseFairseqModel):
    def __init__(self,encoder,vocoder,decoder,tgt_dict,cfg):
        super().__init__()
        self.cfg = cfg
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.encoder=encoder
        self.vocoder=vocoder
        self.decoder=decoder
        #self.proj = nn.Linear(768,128)
        self.softmax = nn.Softmax(dim=2) #for channel dim
        self.spectrogram_extractor = tl.Spectrogram(n_fft=512,hop_length=160)
        self.logmel_extractor = tl.LogmelFilterBank(sr=16000,n_fft=512,n_mels=104)
        self.transpose = torch.nn.ConvTranspose1d(138,277,1)

    @classmethod
    def build_model(cls,cfg:AVHubertCycleConfig,task:FairseqTask):
        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data

        task_pretrain = tasks.setup_task(w2v_args.task)
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])

        encoder_ = task_pretrain.build_model(w2v_args.model)

        encoder = HubertEncoderWrapper(encoder_)
        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            del state['model']['mask_emb']
            #del state['model']['encoder.w2v_model.mask_emb']
            encoder.w2v_model.load_state_dict(state["model"], strict=False)

        encoder.w2v_model.remove_pretraining_modules()

        vocoder = HiFiGANVocoder(cfg.vocoder_path,cfg)
        decoder = encoder
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        return AVHubertCycle(encoder,vocoder,decoder,tgt_dict,cfg)
        
    def stacker(self,feats, stack_order):
        B,T,F = feats.shape
        if T % stack_order != 0:
            res = stack_order - T % stack_order
            res = torch.zeros([B, res, F]).type_as(feats)
            feats = torch.cat([feats, res],dim=1)
        feats = feats.reshape((B,-1, stack_order, F)).reshape(B,-1, stack_order*F)
        return feats

    def forward(self,**kwargs):
        ft = self.freeze_finetune_updates <= self.num_updates
        #with torch.no_grad() if not ft else contextlib.ExitStack():
        output = self.encoder(**kwargs) 

        encoder_output = output['encoder_out'].transpose(0,1) #(7,138,768)
        vocoder_output = self.vocoder(encoder_output).squeeze(1) #B*44160
        sp = self.spectrogram_extractor(vocoder_output) #(B,1,277,257) (B,1,T,F)
        logmel = self.logmel_extractor(sp) # [B,1, T, F] (7,1,277,104)
        #audio_feats = self.stacker(logmel.squeeze(1),2) # [T/stack_order_audio, F*stack_order_audio]
        #print("audio_feats.shape\n",audio_feats.shape) #(7,139,104)

        source_decoder = {"audio": logmel.squeeze(1).transpose(1,2), "video": None}
        decoder_output = self.decoder(source_decoder,None) #(7,277,768)
        decoder_output = decoder_output['encoder_out'].transpose(0,1)
        #print("decoder_output.shape\n",decoder_output.shape)
        B,T,F = encoder_output.shape
        transpose_function = torch.nn.ConvTranspose1d(T,2*T+1,1).cuda()
        encoder_output_upsample = self.softmax(transpose_function(encoder_output.type(torch.float32))) #(7,277,768)

        return encoder_output_upsample,decoder_output

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def get_targets(self,sample, net_output):
        return net_output[1].max(dim=2)

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

if __name__ == '__main__':
    config = AVHubertCycleConfig()
    model = AVHubertCycle.build_model()
