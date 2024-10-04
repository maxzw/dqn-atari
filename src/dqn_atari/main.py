import logging
import os

import hydra
from omegaconf import DictConfig

from dqn_atari import DQN

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def app(cfg: DictConfig) -> None:
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"Output directory: {output_dir}.")

    if cfg.load_path:
        logger.info(f"Loading model from {cfg.load_path}.")
        model = DQN.load(cfg.load_path)
    else:
        model = DQN(**cfg.model)

    if cfg.do_train:
        checkpoint_every = cfg.train.checkpoint_every or cfg.train.training_steps
        num_iterations = cfg.train.training_steps // checkpoint_every
        for i in range(1, num_iterations + 1):
            iter_folder = output_dir + f"/step_{model.steps_trained + checkpoint_every:_}/"
            os.makedirs(iter_folder, exist_ok=True)

            logger.info(f"Training model for {checkpoint_every:_} steps.")
            model.train(
                training_steps=checkpoint_every,
                eval_every=cfg.train.eval_every,
                eval_runs=cfg.train.eval_runs,
            )

            model_path = iter_folder + "model.pt"
            include_buffer = num_iterations == i  # only save in last iteration
            logger.info(f"Saving model to {model_path} `include_buffer`={include_buffer}.")
            model.save(model_path, include_buffer=include_buffer)

            if isinstance(cfg.train.num_gifs, int) and cfg.train.num_gifs > 0:
                gif_path = iter_folder + "gif"
                logger.info(f"Saving GIFs to {gif_path}.")
                model.evaluate(eval_runs=cfg.train.num_gifs, gif_path_format=gif_path)

    if cfg.do_eval:
        result = model.evaluate(**cfg.eval)
        logger.info(f"Test result: {result}.")


if __name__ == "__main__":
    app()
