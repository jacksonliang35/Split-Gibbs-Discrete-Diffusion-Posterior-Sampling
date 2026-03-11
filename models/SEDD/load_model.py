import os
import torch
from .transformer import SEDD
from . import utils
from .ema import ExponentialMovingAverage
from . import graph_lib
from . import noise_lib

from omegaconf import OmegaConf

def load_model_hf(dir, device):
    score_model = SEDD.from_pretrained(dir).to(device)
    graph = graph_lib.get_graph(score_model.config, device)
    noise = noise_lib.get_noise(score_model.config).to(device)
    return score_model, graph, noise


def load_model_local(root_dir, device):
    cfg = utils.load_hydra_config_from_run(root_dir)
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    score_model = SEDD(cfg).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)

    ckpt_dir = os.path.join(root_dir, "checkpoints-meta", "checkpoint.pth")
    loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False)

    score_model.load_state_dict(loaded_state['model'])
    ema.load_state_dict(loaded_state['ema'])

    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    return score_model, graph, noise

def load_model_local_test(root_dir, device):
    print("1. loading config")
    cfg = utils.load_hydra_config_from_run(root_dir)
    print("2. building graph")
    graph = graph_lib.get_graph(cfg, device)
    print("3. building noise")
    noise = noise_lib.get_noise(cfg).to(device)
    print("4. building model")
    score_model = SEDD(cfg).to(device)
    print("5. building ema")
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)

    ckpt_dir = os.path.join(root_dir, "checkpoints-meta", "checkpoint.pth")
    print("6. loading checkpoint:", ckpt_dir)
    loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False)
    print("7. checkpoint keys:", loaded_state.keys())

    print("8. loading model state")
    score_model.load_state_dict(loaded_state["model"])
    print("9. loading ema state")
    ema.load_state_dict(loaded_state["ema"])

    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    score_model.eval()
    print("10. done")

    return score_model, graph, noise

def load_model(root_dir, device):
    return load_model_local_test(root_dir, device)
    # try:
    #     return load_model_local(root_dir, device)
    # except:
    #     return load_model_hf(root_dir, device)
