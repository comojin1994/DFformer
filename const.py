from datasets.shhs import SHHS
from datasets.sleepedf import SleepEDF
from datasets.bci2a import BCIC2a
from datasets.bci2b import BCIC2b


dataset_info_dict = {
    'bci2a': {
        'config': "bci2a_config.yaml",
        'model': "BCIC2a",
        'dataset': BCIC2a,
    },
    'bci2b': {
        'config': "bci2b_config.yaml",
        'model': "BCIC2b",
        'dataset': BCIC2b,
    },
    'sleepedf': {
        'config': "sleepedf_config.yaml",
        'model': "SleepEDF",
        'dataset': SleepEDF,
    },
    'shhs': {
        'config': "shhs_config.yaml",
        'model': "SHHS",
        'dataset': SHHS,
    },
}