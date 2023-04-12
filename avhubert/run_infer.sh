export MASTER_ADDR=localhost
export MASTER_PORT=5678
#fairseq-hydra-train --config-dir conf/pretrain --config-name base_vox_iter5.yaml \
#  task.data=/home/pantianrui/data/lrs3/lrs3/30h_data/ task.label_dir=/home/pantianrui/data/av_hubert/avhubert/clustering/km_feature/ \
#  model.label_rate=25 \
#  hydra.run.dir=ptr_run/pretrain/ common.user_dir=/home/pantianrui/data/av_hubert/avhubert/
  
#HYDRA_FULL_ERROR=1 fairseq-hydra-train --config-dir conf/av-finetune --config-name base_noise_pt_noise_ft_30h.yaml \
# task.data=/home/pantianrui/data/lrs3/lrs3/30h_data/ task.label_dir=/home/pantianrui/data/lrs3/lrs3/30h_data/ \
# task.tokenizer_bpe_model=/home/pantianrui/data/lrs3/lrs3/spm1000/spm_unigram1000.model task.noise_wav=/home/pantianrui/data/musan/tsv/all/\
# model.w2v_path=/home/pantianrui/data/av_hubert/avhubert/checkpoint/base_vox_iter5.pt \
# hydra.run.dir=ptr_run/revise_finetune/ common.user_dir=/home/pantianrui/data/av_hubert/avhubert/




CUDA_VISIBLE_DEVICES=0 python -B infer_revise.py --config-dir ./conf/ --config-name s2s_decode.yaml \
  dataset.gen_subset=test \
  common_eval.path=/home/pantianrui/data/v100_revise_checkpoint/checkpoint_last.pt\
  common_eval.results_path=/home/pantianrui/ReVISE/avhubert/ptr_run/revise/basic_test \
  override.modalities=['audio','video'] common.user_dir=/home/pantianrui/ReVISE/avhubert \
  generation.beam=20 \
  override.data=/home/pantianrui/data/lrs3/lrs3/30h_data/ \
  override.label_dir=/home/pantianrui/data/lrs3/lrs3/30h_data/ \
  override.w2v_path=/home/pantianrui/data/av_hubert/avhubert/checkpoint/base_vox_iter5.pt \
  override.vocoder_path=/home/pantianrui/ReVISE/avhubert/checkpoint/model.pt \
  override.noise_wav=/home/pantianrui/data/musan/tsv/noise/ override.noise_prob=1 override.noise_snr=0 \


