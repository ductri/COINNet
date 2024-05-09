import wandb
from hydra import compose, initialize
import hydra
from omegaconf import OmegaConf
from hydra import compose, initialize

from baselines.cores import phase1
from baselines.cores.phase2 import phase2

@hydra.main(version_base=None, config_path=f"conf", config_name="config_cores")
def main_hydra(conf):
    print(OmegaConf.to_yaml(conf))

    project_name = 'shahana_outlier'
    tags = conf['tags']
    with wandb.init(entity='narutox', project=project_name, tags=tags, config=OmegaConf.to_container(conf, resolve=True)) as run:
        exp_name = run.name
        phase1.main(conf, exp_name)
        # exp_name = 'astral-snowball-889'

        root = conf.root
        method_dir = conf.method_dir
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with initialize(version_base=None, config_path=f"{root}/baselines/cores/phase2/confs", job_name="test_app"):
            phase2_cfg = compose(config_name="resnet34_ins_0.2.yaml", overrides=
                    [f"train_labels={method_dir}/result_dir/{exp_name}/corespairflip0.2train_noisy_labels.npy",
                    f"unsup_idx={method_dir}/result_dir/{exp_name}/corespairflip0.2_noise_pred.npy"])
            print(OmegaConf.to_yaml(phase2_cfg))
            phase2.main(phase2_cfg)


if __name__ == "__main__":
    main_hydra()

