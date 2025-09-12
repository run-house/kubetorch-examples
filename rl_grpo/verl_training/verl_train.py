# # Reinforcement Learning (RL) with verl
# In this example, we will show you how simple it is to launch an RL training with
# `verl` using Kubetorch and Ray. [verl](https://github.com/volcengine/verl) is a popular
# RL training library for large language models (LLMs).
#
# ::youtube[verl]{url="https://youtu.be/-oz49qt_uSM"}
#
# ## Overview
# There are two main components used in this training example:
# * A `run_grpo` function which we will run on a Ray cluster that we bring up in `main()`
# * The `verl` PPO trainer, `run_ppo`, which we will call with our config as-is once all the data and model
# have been downloaded.

import os

import kubetorch as kt
import ray
from download_data import download_data_math, download_model
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict
from verl.trainer.main_ppo import run_ppo

# ## Training Function
# This is the function we will run on remote compute to start the GRPO
# training process. It will use the configuration passed to it (merging with
# the baseline verl config), and we show downloading the data before executing the training.
def run_grpo(cfg):
    GlobalHydra.instance().clear()
    with initialize_config_module(
        config_module="verl.trainer.config", version_base="1.1"
    ):
        base_config = compose(config_name="ppo_trainer")
        with open_dict(base_config):
            cfg = OmegaConf.merge(
                base_config, cfg
            )  # Add our local configs propagating to remote

        download_data_math(
            data_source=cfg.data.hf_data_name,
            train_path=cfg.data.train_files,
            val_path=cfg.data.val_files,
        )
        download_model(
            cfg.actor_rollout_ref.model.hf_model_name, cfg.actor_rollout_ref.model.path
        )

        ray.init(address="auto")
        run_ppo(cfg)


# ## Running with Kubetorch
# We define the main function that sets up the Kubetorch compute environment
# and sends our `run_grpo` function to be executed on the remote compute which is a Ray
# cluster with num nodes and GPUs per node as per our config.
def main(cfg):
    img = (
        kt.Image(
            image_id="verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2"
        )
        .pip_install(["datasets", "omegaconf", "verl"])
        .set_env_vars({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]})
    )

    compute = kt.Compute(
        gpus=cfg.trainer.get("n_gpus_per_node", 1),  # Extract GPU config for kubetorch
        memory="100Gi",
        image=img,
        allowed_serialization=["pickle", "json"],  # Config serialized with pickle
    ).distribute("ray", workers=cfg.trainer.get("nnodes", 2))

    trainer = kt.fn(run_grpo).to(compute)
    trainer(cfg, serialization="pickle")


if __name__ == "__main__":
    verl_config = OmegaConf.load("config.yaml")  # See on GitHub
    main(cfg=verl_config)
