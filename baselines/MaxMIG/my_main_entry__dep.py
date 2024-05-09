import datetime

import wandb

from ours import constants
from baselines.MaxMIG.CIFAR10.main import main


def proxy_entry(config):
    run = wandb.init(project='noisy-labels-tensor', config=config, reinit=True, group='maxmig')
    # If called by wandb.agent, as below,
    # this config will be set by Sweep Controller
    config = wandb.config
    run.tags = config['tags']
    # Each run will have its own unique codename
    codename = datetime.datetime.now()
    wandb.config['codename'] = codename
    wandb.config['model_dir'] = f'{constants.ROOT}/saved_models/maxmig/{codename}/'

    from baselines.MaxMIG.my_config import my_config
    my_config.update(config)
    # if config['dataset_basename'] == 'cifar10':
    #     from baselines.MaxMIG.CIFAR10.main import main
    # else:
    #     raise Exception('basename is wrong')

    main()
    wandb.finish()


