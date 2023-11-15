import os
from typing import Optional, Callable
from glob import glob
from easydict import EasyDict
from models.backbones.dfformer import DFformer
from models.backbones.eegautoencoder import EEGAutoEncoder

model_list = {
    "dfformer": DFformer,
    "eegautoencoder": EEGAutoEncoder,
}


def get_model(args: EasyDict, load_ckpt: Optional[Callable] = None, **kwargs):
    model = model_list[args.model_name](args, **kwargs)

    if load_ckpt is not None:
        if args.WEIGHT_FILENAME is None:
            ckpt_list = sorted(glob(f"{args.CKPT_PATH}/{args.WEIGHT_PATH}/S*.ckpt"))
            args.WEIGHT_FILENAME = ckpt_list[
                args.target_subject
                if args.target_session is None
                else args.target_subject * 3 + args.target_session
            ]
        print(
            "LOG >>> Weight filename: ",
            os.path.join(args.CKPT_PATH, args.WEIGHT_PATH, args.WEIGHT_FILENAME),
        )
        model = load_ckpt(
            model,
            path=os.path.join(args.CKPT_PATH, args.WEIGHT_PATH, args.WEIGHT_FILENAME),
        )

    return model
