import wandb
from hydra import compose, initialize
import hydra
from omegaconf import OmegaConf

from baselines.cores import phase1


@hydra.main(version_base=None, config_path=f"conf", config_name="config_cores")
def main_hydra(conf):
    print(OmegaConf.to_yaml(conf))

    project_name = 'shahana_outlier'
    tags = conf['tags']
    with wandb.init(entity='narutox', project=project_name, tags=tags, config=OmegaConf.to_container(conf, resolve=True)) as run:
        phase1.main(conf, run.name)


if __name__ == "__main__":
    main_hydra()

