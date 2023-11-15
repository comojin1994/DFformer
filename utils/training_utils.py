import os, yaml
from easydict import EasyDict


def get_configs(config_path: str, filename: str, dataset: str):

    with open(os.path.join(config_path, filename)) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        args = EasyDict(config)

        with open(
            os.path.join(config_path, f"model_configs/{args.model_name}.yaml")
        ) as file:
            model_config = yaml.load(file, Loader=yaml.FullLoader)
            args.update(model_config[dataset])

    return args


