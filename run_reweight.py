import wandb
from hydra import compose, initialize
import hydra
from omegaconf import OmegaConf
from baselines.reweight.main import main
from utils import create_ray_wrapper, create_wandb_wrapper


@hydra.main(version_base=None, config_path=f"conf", config_name="config_reweight")
def main_hydra(conf):
    print(OmegaConf.to_yaml(conf))

    runner = create_wandb_wrapper(main)
    if conf.with_ray:
        runner = create_ray_wrapper(runner)
    runner(conf)


if __name__ == "__main__":
    main_hydra()

