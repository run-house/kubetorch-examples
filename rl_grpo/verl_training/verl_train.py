import kubetorch as kt
import ray
from download_data import download_data, download_model
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict
from verl.trainer.main_ppo import run_ppo

# This is the function we will run on remote compute to start the GRPO
# training process. It will use the configuration dictionary passed to it, and
# we show downloading the data before executing the training.
# You likely want to write glue code before launching the verl
# trainer instead of dispatching the trainer directly.
def run_grpo(cfg):
    GlobalHydra.instance().clear()
    with initialize_config_module(
        config_module="verl.trainer.config", version_base="1.1"
    ):
        base_config = compose(config_name="ppo_trainer")  # Grab from verl
        with open_dict(base_config):
            cfg = OmegaConf.merge(
                base_config, cfg
            )  # Add our local configs propagating to remote

        download_data(
            data_source=cfg.data.hf_data_name,
            train_path=cfg.data.train_files,
            val_path=cfg.data.val_files,
        )
        download_model(
            cfg.actor_rollout_ref.model.hf_model_name, cfg.actor_rollout_ref.model.path
        )

        ray.init(address="auto")
        run_ppo(cfg)


def main(cfg):
    img = kt.Image(
        image_id="verlai/verl:base-verl0.5-cu126-cudnn9.8-torch2.7.0-fa2.7.4"
    ).pip_install(["datasets", "omegaconf", "verl", "vllm"])

    # Extract GPU config for kubetorch
    compute = kt.Compute(
        gpus=cfg.trainer.get("n_gpus_per_node", 1),
        image=img,
        allowed_serialization=["pickle", "json"],  # Config serialized with pickle
    ).distribute("ray", workers=cfg.trainer.get("nnodes", 2))

    trainer = kt.fn(run_grpo).to(compute)
    trainer(cfg, serialization="pickle")


if __name__ == "__main__":
    verl_config = OmegaConf.load("config.yaml")
    main(cfg=verl_config)
