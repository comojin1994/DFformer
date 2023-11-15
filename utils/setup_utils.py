import torch


def get_device(GPU_NUM: str) -> torch.device:
    if torch.cuda.device_count() == 1:
        output = torch.device(f'cuda:{GPU_NUM}')
    elif torch.cuda.device_count() > 1:
        output = torch.device(f'cuda')
    else:
        output = torch.device('cpu')

    print(f'{output} is checked')
    return output