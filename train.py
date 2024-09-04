import argparse
import os
import tempfile
import time
import numpy as np
from engine import Engine
from utils import config_util
from utils.general_util import log, get_temp_dir, mkdir_p
import pandas as pd
from pdb import set_trace

def main(args):
    resume = False
    if args.writer:
        filename = args.writer
    else:
        filename = "testing"
    node = os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else "CPU"
    node_flag = node not in ("CPU")
    # If tuning
    if args.tune:
        assert os.path.exists(args.config_path)
        if args.save_dir:
            save_dir = mkdir_p(args.save_dir)
        resume = True
        # Load from configs from the path given
        configs = config_util.load_configs(args.config_path)

        # Save the configs in ${save_dir}
        config_util.save_configs(args.config_path, os.path.join(save_dir, "configs.yaml"))
    else:
        if not args.config_path:
            # resume the previous training
            assert os.path.exists(args.save_dir)
            resume = True
            save_dir = args.save_dir
            configs = config_util.load_configs(os.path.join(save_dir, "configs.yaml"))
        else:
            configs = config_util.load_configs(args.config_path)
            if args.save_dir:
                # start a new training from a specific derectory
                if not os.path.exists(args.save_dir):
                    save_dir = mkdir_p(args.save_dir)
                else:
                    save_dir = args.save_dir
            else:
                # start a new training from an automatically generated directory
                root_dir = get_temp_dir()

                data_names = sorted(set([
                    name for name, split in configs["data"]["split"]["train"]
                ]))
                tempfile.tempdir = mkdir_p(
                    os.path.join(root_dir, "+".join(data_names).upper())
                )

                model_name = configs["model"]["name"]
                if model_name == "fusion":
                    model_name = "fusion_{}+{}".format(
                        configs["model"]["first"]["name"],
                        configs["model"]["second"]["name"]
                    )

                
                
                train_prefix = "%s-%s-%s-" % (
                    model_name.upper(),
                    time.strftime("%Y%m%d-%H%M%S"),
                    node
                )
                save_dir = tempfile.mkdtemp(
                    suffix="-" + args.tag if args.tag else None,
                    prefix=train_prefix
                )
            config_util.save_configs(args.config_path, os.path.join(save_dir, "configs.yaml"))
            

    log.infov("Working Directory: {}".format(save_dir))

    
    datetime = filename.split('/')[-1].lstrip(node + "-") if filename not in ("testing") else time.strftime("%Y%m%d-%H%M%S")

    if not os.path.isfile("experiments.csv"):
        df = pd.DataFrame(
            columns=["Working Directory", "Output File", "datetime", "node"]
        )
    else:
        df = pd.read_csv("experiments.csv")

    if not df["Working Directory"].isin([save_dir]).any():    

        row = {
            "Working Directory": save_dir, 
            "Output File": filename,
            "datetime": datetime,
            "node": node
        }
        
        df.loc[len(df)] = row

    else:
        
        if not df["Output File"].isin([filename]).any():
            df.loc[df["Working Directory"] == args.save_dir, "Output File"] = args.writer
        
        df.loc[df["Working Directory"] == args.save_dir, "datetime"] = datetime

        if not node_flag:
            df.loc[df["Working Directory"] == args.save_dir, "node"] = 0
        else:
            df.loc[df["Working Directory"] == args.save_dir, "node"] = int(node)

    df.to_csv("experiments.csv", index=False)

    engine = Engine(
            mode="train", configs=configs, save_dir=save_dir
        )
    
    

    engine.train(validate=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="",
                        help="path to a config")
    parser.add_argument("--save_dir", default="",
                        help="directory to save checkpointables")
    parser.add_argument("--tag", default="",
                        help="tag to discern training results")
    parser.add_argument("--tune", default=False, type=bool, 
                        help="the transfer learning parameter")
    parser.add_argument("--ray_tune", default=False, type=bool,
                        help="fine-tune the hyperparameters")
    parser.add_argument("--writer", type=str, default="",
                        help="Output location of slurm job")
    args = parser.parse_args()

    main(args)
