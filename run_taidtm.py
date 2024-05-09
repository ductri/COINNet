import wandb
from hydra import compose, initialize
import hydra
from omegaconf import OmegaConf

from baselines.taidtm import common_DNNTM


@hydra.main(version_base=None, config_path=f"conf", config_name="config_taidtm")
def main_hydra(conf):
    print(OmegaConf.to_yaml(conf))

    project_name = 'shahana_outlier'
    with wandb.init(entity='narutox', project=project_name, tags=conf.tags, config=OmegaConf.to_container(conf, resolve=True)) as run:
        # common_DNNTM.set_config(conf, run.name)

        from baselines.taidtm.train import super_main
        super_main(conf, run.name)


if __name__ == "__main__":
    main_hydra()

