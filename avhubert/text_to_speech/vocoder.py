# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig
from fairseq.models import BaseFairseqModel, register_model
#from codehifigan import CodeGenerator as CodeHiFiGANModel
from .hifigan import Generator as HiFiGANModel
from .hub_interface import VocoderHubInterface
import pdb
logger = logging.getLogger(__name__)


class HiFiGANVocoder(nn.Module):
    def __init__(
        self, checkpoint_path: str, model_cfg: Dict[str, str], fp16: bool = False
    ) -> None:
        super().__init__()
        self.model = HiFiGANModel(model_cfg)
        state_dict = torch.load(checkpoint_path)
        self.model.load_state_dict(state_dict["generator"],strict=False)
        if fp16:
            self.model.half()
        logger.info(f"loaded HiFiGAN checkpoint from {checkpoint_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B x) T x D -> (B x) 1 x T
        model = self.model.train()
        if len(x.shape) == 2:
            return model(x.unsqueeze(0).transpose(1, 2)).squeeze(0)
        else:
            return model(x.transpose(-1, -2))

    @classmethod
    def from_data_cfg(cls, args, data_cfg: S2TDataConfig):
        vocoder_cfg = data_cfg.vocoder
        assert vocoder_cfg.get("type", "griffin_lim") == "hifigan"
        with open(vocoder_cfg["config"]) as f:
            model_cfg = json.load(f)
        return cls(vocoder_cfg["checkpoint"], model_cfg, fp16=args.fp16)


@register_model("CodeHiFiGANVocoder")
class CodeHiFiGANVocoder(BaseFairseqModel):
    def __init__(
        self, checkpoint_path: str, model_cfg: Dict[str, str], fp16: bool = False
    ) -> None:
        super().__init__()
        self.model = CodeHiFiGANModel(model_cfg)
        if torch.cuda.is_available():
            state_dict = torch.load(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict["generator"])
        self.model.eval()
        if fp16:
            self.model.half()
        self.model.remove_weight_norm()
        logger.info(f"loaded CodeHiFiGAN checkpoint from {checkpoint_path}")

    def forward(self, x: Dict[str, torch.Tensor], dur_prediction=False) -> torch.Tensor:
        assert "code" in x
        x["dur_prediction"] = dur_prediction

        # remove invalid code
        mask = x["code"] >= 0
        x["code"] = x["code"][mask].unsqueeze(dim=0)
        if "f0" in x:
            f0_up_ratio = x["f0"].size(1) // x["code"].size(1)
            mask = mask.unsqueeze(2).repeat(1, 1, f0_up_ratio).view(-1, x["f0"].size(1))
            x["f0"] = x["f0"][mask].unsqueeze(dim=0)

        return self.model(**x).detach().squeeze()

    @classmethod
    def from_data_cfg(cls, args, data_cfg):
        vocoder_cfg = data_cfg.vocoder
        assert vocoder_cfg is not None, "vocoder not specified in the data config"
        with open(vocoder_cfg["config"]) as f:
            model_cfg = json.load(f)
        return cls(vocoder_cfg["checkpoint"], model_cfg, fp16=args.fp16)

    @classmethod
    def hub_models(cls):
        base_url = "http://dl.fbaipublicfiles.com/fairseq/vocoder"
        model_ids = [
            "unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur",
            "unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10_dur",
            "unit_hifigan_HK_layer12.km2500_frame_TAT-TTS",
        ]
        return {i: f"{base_url}/{i}.tar.gz" for i in model_ids}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        config="config.json",
        fp16: bool = False,
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            config_yaml=config,
            fp16=fp16,
            is_vocoder=True,
            **kwargs,
        )

        with open(f"{x['args']['data']}/{config}") as f:
            vocoder_cfg = json.load(f)
        assert len(x["args"]["model_path"]) == 1, "Too many vocoder models in the input"

        vocoder = CodeHiFiGANVocoder(x["args"]["model_path"][0], vocoder_cfg)
        return VocoderHubInterface(vocoder_cfg, vocoder)


def get_vocoder(args, data_cfg: S2TDataConfig):
    if args.vocoder == "griffin_lim":
        return GriffinLimVocoder.from_data_cfg(args, data_cfg)
    elif args.vocoder == "hifigan":
        return HiFiGANVocoder.from_data_cfg(args, data_cfg)
    elif args.vocoder == "code_hifigan":
        return CodeHiFiGANVocoder.from_data_cfg(args, data_cfg)
    else:
        raise ValueError("Unknown vocoder")

if __name__ == '__main__':
    checkpoint_path = '/home/pantianrui/data/av_hubert/avhubert/checkpoint/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur/model.pt'
    cfg = {}
    cfg['resblock_kernel_sizes'] = [3,7,11]
    cfg['upsample_rates'] = [5,4,4,2,2]
    cfg['model_in_dim'] = 128
    cfg['upsample_initial_channel'] = 512
    cfg['upsample_kernel_sizes'] = [11,8,8,4,4]
    cfg['resblock_dilation_sizes'] = [[1,3,5], [1,3,5], [1,3,5]]
    model = HiFiGANVocoder(checkpoint_path,cfg)
    pdb.set_trace()


