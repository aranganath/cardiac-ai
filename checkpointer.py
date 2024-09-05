# Reference: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/checkpoint.py
import copy
import os
import torch

from fvcore.common.checkpoint import (Checkpointer, _strip_prefix_if_present,
                                      _IncompatibleKeys)
from utils.general_util import log


class Checkpointer(Checkpointer):
    def __init__(self, mode, model, save_dir, default_name="model",
                 **checkpointables):
        self.mode = mode
        self.default_name = default_name
        super().__init__(
            model,
            save_dir,
            **checkpointables
        )

    def save(self, name=None, **kwargs):
        if name is None:
            name = self.default_name
        super().save(name, **kwargs)

    def get_checkpoint_path(self):
        basename = "{}.pth".format(self.default_name)
        save_path = os.path.join(self.save_dir, basename)
        return save_path

    def resume_or_load(self, path=None, resume=True):
        if isinstance(path, dict):
            assert not resume
            return self.load_multiple_checkpoints(path)

        if resume:
            if (not path) or (not os.path.exists(path)):
                path = self.get_checkpoint_path()
            return self.load(path)
        else:
            if self.mode == "test":
                path = self.get_checkpoint_path()
            return self.load(path, checkpointables=[])

    def load_multiple_checkpoints(self, paths):
        merged_state_dict = {}
        for name, path in paths.items():
            if os.path.exists(path):
                checkpoint = self._load_file(path)
                checkpoint_state_dict = checkpoint.pop("model")
                _strip_prefix_if_present(checkpoint_state_dict, "module.")
                state_dict = add_prefix(
                    checkpoint_state_dict,
                    prefix="{}_model".format(name)
                )
                merged_state_dict.update(state_dict)

        if not merged_state_dict:
            log.warn("No checkpoint found. Initializing model from scratch")
            return {}

        incompatible = self.model.load_state_dict(merged_state_dict, strict=False)
        if incompatible is not None:
            self._log_incompatible_keys(
                _IncompatibleKeys(
                    missing_keys=incompatible.missing_keys,
                    unexpected_keys=incompatible.unexpected_keys,
                    incorrect_shapes=[]
                )
            )
        return {}

    def _load_file(self, filename):
        log.info("Loading from {}...".format(filename))
        loaded = super()._load_file(filename)
        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded


def add_prefix(weights, prefix=""):
    log.infov("  Remapping the original weights ......")
    original_keys = sorted(weights.keys())
    new_weights = {}
    for orig in original_keys:
       new_weights[prefix + "." + orig] = weights[orig]
    return new_weights
