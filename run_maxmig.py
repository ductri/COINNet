import wandb
from hydra import compose, initialize
import hydra
from omegaconf import OmegaConf

from utils import create_ray_wrapper, create_wandb_wrapper
from baselines.MaxMIG.global_conf import set_conf


def maxmig_main(conf, unique_name):
    conf.model_dir = f'{conf.model_dir}/{unique_name}/'
    print('xxxxx')
    print(f'model_dir={conf.model_dir} \n\n\n')
    set_conf(conf)

    from baselines.MaxMIG.CIFAR10.main import main
    main()


@hydra.main(version_base=None, config_path=f"conf", config_name="config_maxmig")
def main_hydra(conf):
    print(OmegaConf.to_yaml(conf))

    runner = create_wandb_wrapper(maxmig_main)
    if conf.with_ray:
        runner = create_ray_wrapper(runner)
    runner(conf)


if __name__ == "__main__":
    main_hydra()

