import numpy as np
import os
import torch

from collections import OrderedDict
from torch import optim
from torch.utils.data import ConcatDataset, random_split

# import datasets
import datasets.dataset
import metrics
import losses
import models.TransfomerModel
import schedulers
from datasets import dataset
from datasets.base import CustomDataLoaders
# from datasets import pdbbindLP, pdbbind, mpro, gmd, dengue, zika, westnile
# from datasets.base import CustomDataLoaders
# from models import sgcnn, conv3d_net, pcn, fusion, egnn, CombativeEGNN, CCEGNN, geomFusion, CascadeModels, efficient_se3, nonefficient_se3
from models import TransfomerModel
import models
from utils.general_util import log
from pdb import set_trace
import shutil

SEED = 22

DATASETS = {
    "ECG": dataset.TransformerDataset,
}

MODELS = {
    "Transformer": TransfomerModel.TimeSeriesTransformer,
    "SelfTransformer": TransfomerModel.SelfTimeSeriosTransformer
}

OPTIMIZERS = {
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
    "adam": optim.Adam,
}

SCHEDULERS = {
    "step": schedulers.StepLR,
    "multi_step": schedulers.MultiStepLR,
    "polynomial": schedulers.PolynomialLR,
    "cosine": schedulers.CosineAnnealing,
}

LOSSES = {
    "l2": losses.L2Loss,
    "mse": losses.CustomMSELoss,
    "pmse": losses.CustomMSELoss,
    "bce": losses.BCELoss,
    "huber": losses.HuberLoss
}

METRICS = {
    "r2": metrics.R2,
    "mae": metrics.MAE,
    "mse": metrics.MSE,
    "rmse": metrics.RMSE,
    "pearson_r": metrics.PearsonR,
    "spearman_r": metrics.SpearmanR,
}



def build_dataloader(configs, mode="train"):
    log.infov("Begin building dataset")
    dataset_dict = {}
    if mode == "train":
        train_sets = configs["data"]["split"]["train"]
        dataset_dict["train"] = ConcatDataset([
            DATASETS[data_name](mode=mode, subset=subset, configs=configs["data"])
            for data_name, subset in train_sets
        ])

        if "val" in configs["data"]["split"]:
            val_sets = configs["data"]["split"]["val"]
            dataset_dict["val"] = ConcatDataset([
                DATASETS[data_name](mode=mode, subset=subset, configs=configs["data"])
                for data_name, subset in val_sets
            ])
        else:
            generator = torch.Generator()
            generator.manual_seed(SEED)

            train_dataset = dataset_dict["train"]
            if "ratio" in configs["data"]:
                train_size = int(configs["data"]["ratio"] * len(train_dataset))
            else:
                train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size

            train_dataset, val_dataset = random_split(
                train_dataset,
                [train_size, val_size],
                generator=generator,
            )
            dataset_dict["train"] = train_dataset
            dataset_dict["val"] = val_dataset
    else:
        test_sets = configs["data"]["split"]["test"]
        dataset_dict["test"] = ConcatDataset([
            DATASETS[data_name](mode=mode, subset=subset, configs=configs["data"])
            for data_name, subset in test_sets
        ])
    dataloaders = CustomDataLoaders(
        dataset_dict=dataset_dict,
        configs=configs[mode],
    )
    return dataloaders


def build_loss(loss_configs):
    loss_dict = {}
    for loss_config in loss_configs:
        loss_name = loss_config.pop("name")
        loss_dict[loss_name] = LOSSES[loss_name](**loss_config)
    return loss_dict


def build_model(model_configs, data_configs):
    model_name = model_configs["name"]
    
    if model_name in ("fusion","geomfusion"):
        model_params = model_configs.pop("params", {})
        models = {
            "first": (
                model_configs["first"]["name"],
                build_model(model_configs["first"], data_configs)
            ),
            "second": (
                model_configs["second"]["name"],
                build_model(model_configs["second"], data_configs)
            )
        }
        loss = build_loss(model_configs["loss"])

        model = MODELS[model_name](
            in_channels=None,
            out_channels=1,
            loss_fn=loss,
            models=models,
            **model_params
        )
        return model

    model_params = model_configs["params"]
    if model_name in ("pcn"):
        if "max_atoms" in data_configs:
            model_params["max_atoms"] = data_configs["max_atoms"]

    # TODO(june): set `in_channels` automatically using dataloader.
    data_names = []
    for _, data in data_configs["split"].items():
        data_names += [name for name, subset in data]

    
    loss = build_loss(model_configs["loss"]) if "loss" in model_configs else None
    model = MODELS[model_name](
        in_channels=12,
        out_channels=75,
        loss_fn=loss,
        **model_params,
    )
    return model


def build_optimizer(optim_configs, model):
    optim_name = optim_configs.get("name", "adam")
    optim_params = model.get_optimizer_params(optim_configs["params"])
    if optim_name in OPTIMIZERS:
        optimizer = OPTIMIZERS[optim_name](**optim_params)
    else:
        raise ValueError("Invalid optimizer: {}".format(optim_name))
    return optimizer


def build_scheduler(scheduler_configs, optimizer):
    scheduler_name = scheduler_configs.get("name", "multi_step")
    scheduler_params = scheduler_configs.get("params", {})
    scheduler_params["optimizer"] = optimizer

    if scheduler_name in SCHEDULERS:
        scheduler = SCHEDULERS[scheduler_name](**scheduler_params)
    else:
        raise ValueError("Invalid scheduler: {}".format(scheduler_name))
    return scheduler


def build_eval_meter(eval_configs):
    metric_names = eval_configs.get("metrics", [])

    metric_dict = {}
    for metric_name in metric_names:
        if metric_name in METRICS:
            metric_dict[metric_name] = METRICS[metric_name]()
        else:
            raise ValueError("Invalid metric: {}".format(metric_name))

    main_metric = eval_configs["main_metric"] \
        if "main_metric" in eval_configs \
        else eval_configs["metrics"][0]

    return metrics.Meter(metric_dict, main_metric)
