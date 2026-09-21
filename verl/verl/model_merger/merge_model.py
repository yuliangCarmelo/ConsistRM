from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger

config = ModelMergerConfig(
    operation="merge",
    backend="fsdp",
    local_dir="./../../../checkpoints/verl_grpo_format/qwen3_14b_format_1024_bs64_ro8_round/global_step_110/actor",
    target_dir="./../../../save_models/qwen3_14b_format_1024_bs64_ro8_round_GRPO_step_110",
    hf_model_config_path="./../../../checkpoints/verl_grpo_format/qwen3_14b_format_1024_bs64_ro8_round/global_step_110/actor/huggingface"  # 假设配置文件在该路径下
)

merger = FSDPModelMerger(config)
merger.merge_and_save()
merger.cleanup()