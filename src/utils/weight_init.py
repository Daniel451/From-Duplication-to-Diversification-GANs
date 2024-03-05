import torch


def init_weights(m, weight_init):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
        if weight_init == "normal":
            torch.nn.init.normal_(m.weight)
        elif weight_init == "xavier":
            torch.nn.init.xavier_uniform_(m.weight)
        elif weight_init == "kaiming":
            torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1.0)  # Gamma initialized to 1
        torch.nn.init.constant_(m.bias, 0.0)  # Beta initialized to 0
