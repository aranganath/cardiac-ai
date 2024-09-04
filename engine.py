import json
import numpy as np
import os
import time
import torch

from collections import OrderedDict
import torch.distributed as dist
# from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
# from fvcore.common.checkpoint import Checkpointer

from builder import (build_dataloader, build_model, build_optimizer,
                     build_scheduler,  build_eval_meter)
# from builder import (build_dataloader,, ,
                    #  , )
from checkpointer import Checkpointer
from utils.general_util import log, mkdir_p
from pdb import set_trace
import subprocess as sp

class Engine(object):
    def __init__(self, mode, configs, save_dir, resume=False):
        self.mode = mode
        self.configs = configs
        self.save_dir = save_dir
        self.resume = resume
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if device == "cpu":
            log.warn("GPU is not available.")
        else:
            log.warn("Total of {} GPU(s) is/are available.".format(torch.cuda.device_count()))
        self.dataloaders = build_dataloader(configs, mode=mode)
        
        self.model = build_model(configs["model"], configs["data"])

        if torch.cuda.device_count() > 0:
            
            log.infov("Loading {} parameters".format(sum([parameter.numel() for parameter in self.model.parameters()])))
            
            self.model = torch.nn.DataParallel(self.model) # .float()
            self.module = self.model.module
        else:
            log.infov("Loading {} parameters".format(sum([parameter.numel() for parameter in self.model.parameters()])))
            self.module = self.model
            
        self.model.to(self.device)

        # Build an optimizer and loss
        checkpointables = {}
        if mode == "train":
            self.optimizer = build_optimizer(
                self.configs["train"]["optimizer"], self.module
            )
            self.scheduler = build_scheduler(
                self.configs["train"]["scheduler"], self.optimizer
            )
            checkpointables = {
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
            }
        self.eval_meter = build_eval_meter(self.configs["test"])

        # Build a checkpointer
        default_name = "model"
        if mode == "test":
            if ("checkpoint" in configs["test"]) and (configs["test"]["checkpoint"] == "best"):
                default_name = "best_model"
        self.checkpointer = Checkpointer(
            mode, self.model, save_dir, default_name, **checkpointables
        )

        # Specify output directory
        if (mode == "test") and ("test" in configs["data"]["split"]):
            test_names = sorted(set([
               name+ "_" + split for name, split in configs["data"]["split"]["test"]
            ]))
            test_name = "+".join(test_names)
            self.output_dir = mkdir_p(os.path.join(save_dir, test_name))

        else:
            self.output_dir = save_dir
        
        
    def train(self, validate=True):
        
        ckpt_path = None
        # Initialize with a pretrained model.
        if not self.resume:
            if "ckpt_path" in self.configs["model"]:
                ckpt_path = self.configs["model"]["ckpt_path"]
            else:
                if self.configs["model"]["name"] == "fusion":
                    first_configs = self.configs["model"]["first"]
                    second_configs = self.configs["model"]["second"]
                    ckpt_path = {
                        "first": first_configs["ckpt_path"] if "ckpt_path" in first_configs else "",
                        "second": second_configs["ckpt_path"] if "ckpt_path" in first_configs else "",
                    }
            
        checkpoint = self.checkpointer.resume_or_load(path=self.configs["model"]["ckpt_path"], resume=self.resume)

        start_epoch, self.best_epoch = 0, 0
        self.best_metric = self.eval_meter.initial_metric
        if self.resume:
            if "epoch" in checkpoint:
                start_epoch = int(checkpoint["epoch"])
            if "best_epoch" in checkpoint:
                self.best_epoch = int(checkpoint["best_epoch"])
            if "best_metric" in checkpoint:
                self.best_metric = checkpoint["best_metric"]

        num_epochs = self.configs["train"].get("epochs", 100)
        step, next_checkpoint_step = 0, self.configs["train"].get("checkpoint_step", 100)

        self.module.initialize()
        self.print_params()
        log.info(
            "Train for {} epochs starting from epoch {}".format(num_epochs, start_epoch))

        for epoch in range(start_epoch, num_epochs):
            train_start = time.time()
            train_loss, step, next_checkpoint_step =\
                self._train_one_epoch(epoch, step, next_checkpoint_step)
            train_time = time.time() - train_start

            log.infov("[Epoch {:03d}] Training completed in {:.2f} sec".format(epoch, train_time))
            log.warn("[Epoch {:03d}] (Overall) Loss={:.4f}".format(epoch, train_loss))

            if (self.scheduler is not None) and self.scheduler.by_epoch:
                self.scheduler.step()

            if validate:
                val_start = time.time()
                self.eval_meter.reset()
                self.validate(epoch)
                val_time = time.time() - val_start

                log.infov("[Epoch {:03d}] Validation completed in {:.2f} sec".format(epoch, val_time))

                results, eval_strs = {}, []
                metric_vals = self.eval_meter.compute()
                for k, v in metric_vals.items():
                    eval_strs.append(" {}: {:.4f} ".format(str(k), v))
                    if not isinstance(v, float):
                        v = v.astype(np.float64)
                    results[k] = v
                log_msg = "[Epoch {:03d}] (Overall)".format(epoch) + "|".join(eval_strs)
                log.warn(log_msg)

                with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                    f.write(log_msg + "\n")

                if self.eval_meter.is_better(results, self.best_metric):
                    log.warn("            Break the best {}: {:.4f} -> {:.4f}".format(
                        self.eval_meter.main_metric,
                        self.best_metric,
                        results[self.eval_meter.main_metric]
                    ))
                    # self.writer.add_scalar('Loss/Best MSE (eval)', results[self.eval_meter.main_metric], epoch)
                    self.best_epoch = epoch
                    self.best_metric = results[self.eval_meter.main_metric]
                    model_name = "best_model"
                    self.checkpointer.save(
                        name=model_name,
                        model_state_dict=self.model.state_dict(),
                        optimizer_state_dict=self.optimizer.state_dict(),
                        best_epoch=self.best_epoch,
                        best_metric=self.best_metric,
                        step=step,
                        epoch=epoch
                    )
                    results["epoch"] = epoch
                    with open(os.path.join(self.output_dir, "valid_results.json"), "w") as f:
                        json.dump(results, f, indent=4)

        end_msg = "Training ends! Best {} (Epoch {:03d}) = {:.4f}".format(
            self.eval_meter.main_metric, self.best_epoch, self.best_metric
        )
        log.warn(end_msg)
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(end_msg)
        log.info("Output directory: {}".format(self.output_dir))    

    def print_params(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                log.info("  {:60}: {}".format(name, param.shape))
            else:
                log.infov("  {:60}: {}".format(name, param.shape))

    def validate(self, epoch):
        dataloader = self.dataloaders.get("val")
        num_batches = len(dataloader)
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                inputs = self.to_device(batch)
                labels = inputs["y"]

                results = self.model(inputs)
                metric_vals = self.eval_meter.update(results, labels)

                batch_str = "[Epoch {:03d}] ({}/{})".format(epoch, i, num_batches - 1)
                eval_strs = []
                for k, v in metric_vals.items():
                    eval_strs.append(" {}: {:.4f} ".format(str(k), v))
                log.info(batch_str + "|".join(eval_strs))

                self.optimizer.zero_grad()

        torch.cuda.empty_cache()
        return results

    def _train_one_epoch(self, epoch, step, next_checkpoint_step):
        dataloader = self.dataloaders.get("train")
        
        num_batches = len(dataloader)
        losses = []

        self.model.train()
        
        for i, batch in enumerate(dataloader):
            
            inputs = self.to_device(batch)
            loss_dict = self.model(inputs)
            loss = sum(loss_dict.values()).mean()
            loss_val = loss.cpu().data.item()

            if loss.isnan():
                print('NaN encountered')
                exit(0)
            losses.append(loss_val)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            
            
            if (self.scheduler is not None) and (not self.scheduler.by_epoch):
                self.scheduler.step()

            # Loss
            loss_str = "[Epoch {:03d} | LR={:.7f}] ({}/{}) Loss={:.4f} | ".format(
                epoch, self.scheduler.get_last_lr()[0], i, num_batches - 1, loss_val)
            loss_str += " | ".join(["{:s}={:.4f}".format(
                _type, _value.mean().item()) for _type, _value in loss_dict.items()])
            log.info(loss_str)

            # Save checkpoint
            step += 1
            if step >= next_checkpoint_step:
                self.checkpointer.save(
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    loss=loss,
                    step=step,
                    epoch=epoch,
                    best_epoch=self.best_epoch,
                    best_metric=self.best_metric,
                )
                next_checkpoint_step += self.configs["train"].get("checkpoint_step", 100)
            
            
        torch.cuda.empty_cache()
            
        return np.mean(losses), step, next_checkpoint_step
    
    def to_device(self, items):

        if isinstance(items, (dict)):
            for k, v in items.items():
                
                if isinstance(v, (str, int)):
                    pass 
                else:
                    items[k] = v.float().to(self.device)

        elif isinstance(items, tuple):
            items = tuple(x.to(self.device).float() for x in items)
        elif isinstance(items, list):
            items = tuple(x.to(self.device).float() for x in items)
        elif isinstance(items, (str,int)):
            pass
        else:
            items = items.float().to(self.device)
        return items
    
    def evaluate(self):
        checkpoint = self.checkpointer.resume_or_load(resume=False)
        dataloader = self.dataloaders.get("test")
        num_batches = len(dataloader)
        self.model.eval()

        predictions = {}
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                inputs = self.to_device(batch)
                labels = inputs['y']
                results = self.model(inputs)
                predictions.update({
                    str(self.configs["data"]["file"] + i*self.configs["test"]["batch_size"] + j): (r.cpu().numpy().tolist(), b.cpu().numpy().tolist()) 
                    for j, (b, r) in enumerate(zip(batch["y"], results))
                })

                metric_vals = self.eval_meter.update(results, labels)

                # Compute evaluation metrics
                batch_str = "[Epoch {:03d}] ({}/{})".format(checkpoint["epoch"], i, num_batches - 1)
                eval_strs = []
                for k, v in metric_vals.items():
                    eval_strs.append(" {}: {:.4f} ".format(str(k), v))
                log.info(batch_str + "|".join(eval_strs))

        results, eval_strs = {}, []
        metric_vals = self.eval_meter.compute()
        for k, v in metric_vals.items():
            eval_strs.append(" {}: {:.4f} ".format(str(k), v))
            if not isinstance(v, float):
                v = v.astype(np.float64)
            results[k] = v
        log_msg = "[Epoch {:03d}] (Overall)".format(checkpoint["epoch"]) + "|".join(eval_strs)
        log.warn(log_msg)

        results["epoch"] = checkpoint["epoch"]
        with open(os.path.join(self.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

        with open(os.path.join(self.output_dir, "predictions.json"), "w") as f:
            json.dump(predictions, f, indent=4)


# %%
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.set_default_device(device)

# # %%
# torch.set_default_dtype(torch.float64)
# # Okay, we kind of have a way of loading the data
# # We need to collect the data and feed it to the transformer model
# # Now how do we that ?


# # Torch Tensor data !

# # Now, also get the activation times
# path = './Datasets/intracardiac_dataset/'
# train_test_ratio = 0.8
# VmTrainData, pECGTrainData, VmDataTest, pECGTestData, actTimeTrain, actTimeTest  = fileReader(path, 20, train_test_ratio)
# print('Data loading from files - complete')

# VmTrainData = (VmTrainData - torch.min(VmTrainData))/(torch.max(VmTrainData)-torch.min(VmTrainData))
# pECGTrainData = (pECGTrainData - torch.min(pECGTrainData))/(torch.max(pECGTrainData) - torch.min(pECGTrainData))

# VmDataTest = (VmDataTest - torch.min(VmDataTest))/(torch.max(VmDataTest) - torch.min(VmDataTest))

# pECGTestData = (pECGTestData - torch.min(pECGTestData))/(torch.max(pECGTestData) - torch.min(pECGTestData))
# print('Normalization - complete!')

# # %%
# dim_val = 75
# n_heads = 75
# n_decoder_layers = 1
# n_encoder_layers = 1
# input_size = 12
# dec_seq_len = 498
# enc_seq_len = 500

# max_seq_len = enc_seq_len
# train_batch_size = 20
# test_batch_size = 1
# batch_first= True
# output_size = 75
# window_size = 75
# stride = window_size
# output_sequence_length = window_size

# # %%

# # Get the indices of the sequences
# # The idea is: start - stop, where stop - start is window_size
# # This means, each tuple in VmInd and pECGInd is 50 steps
# datInd = get_indices_entire_sequence(VmData = VmTrainData, 
#                                     ECGData = pECGTrainData,
#                                     window_size= window_size, 
#                                     step_size = stride
#                                 )

# # Now let's collect the training data in the Transformer Dataset class
# TrainData = TransformerDataset(VmData = VmTrainData,
#                                     datInd=datInd,
#                                     ECGData = pECGTrainData,
#                                     actTimeData=actTimeTrain,
#                                     enc_seq_len = enc_seq_len,
#                                     dec_seq_len = dec_seq_len,
#                                     target_seq_len = output_sequence_length
#                                 )


# TrainData = DataLoader(TrainData, batch_size=train_batch_size, shuffle=False, generator=torch.Generator(device='cuda'))
# print(TrainData.batch_size)

# datInd = get_indices_entire_sequence(VmData = VmDataTest, 
#                                             ECGData = pECGTestData, 
#                                             window_size= window_size, 
#                                             step_size = stride)


# # Now, let's load the test data
# TestData = TransformerDataset(
#                             VmData = VmDataTest, 
#                             ECGData = pECGTestData,
#                             actTimeData=actTimeTest,
#                             datInd=datInd,
#                             enc_seq_len = enc_seq_len,
#                             dec_seq_len = dec_seq_len,
#                             target_seq_len = output_sequence_length
#             )

# TestData = DataLoader(TestData, test_batch_size, shuffle=False, generator=torch.Generator(device='cuda'))

# # %%

# model = TimeSeriesTransformer(
#     dim_val=dim_val,
#     batch_first=batch_first,
#     input_size=input_size, 
#     dec_seq_len=dec_seq_len,
#     out_seq_len=output_sequence_length, 
#     n_decoder_layers=n_decoder_layers,
#     n_encoder_layers=n_encoder_layers,
#     n_heads=n_heads,
#     num_predicted_features=output_size
# )

# print('Total Model parameters:', sum([parameter.numel() for parameter in model.parameters()]))


# # Define the MSE loss
# criterion = torch.nn.HuberLoss(delta=1)

# # Define cross-entropy loss for the activation times
# criterion2 = torch.nn.MSELoss()

# # Define the optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.999)

# EPOCHS= 80000
# train_losses = []
# src_mask = generate_square_subsequent_mask(
#                 dim1=output_sequence_length,
#                 dim2=6
#             )
# tgt_mask = generate_square_subsequent_mask(
#     dim1=output_sequence_length,
#     dim2=6
# )

# train_interval = 100
# model_interval = 10000
# pbar = tqdm(range(EPOCHS), desc='Training')

# for epoch in pbar:
#     PATH = ''
#     running_loss = 0
#     for i, (src, trg, trg_y, act_time) in enumerate(TrainData):
#         optimizer.zero_grad()
#         recon, activation = model(
#             src=src.permute(0,2,1).to(device),
#             tgt=trg.to(device),
#             src_mask=src_mask,
#             tgt_mask=tgt_mask
#         )
#         y = torch.cat([trg[:,0,:,:].unsqueeze(1) , trg_y], axis = 1)
#         loss = criterion(recon.to(device), y.to(device)) + criterion2(activation.to(device), act_time.type(torch.float64).to(device))
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#         running_loss += loss.item()
#         if (epoch+1) % train_interval == 0:
#             model.train = False
#             with torch.no_grad():
#                 src, trg, trg_y, act_time = next(iter(TestData))
#                 recon, activation = model(
#                     src=src.permute(0,2,1).to(device),
#                     tgt=trg.to(device),
#                     src_mask=src_mask,
#                     tgt_mask=tgt_mask
#                 )
#                 y = torch.cat([trg[:,0,:,:].unsqueeze(1) , trg_y], axis = 1)
#                 row = 7
#                 column = 10

#                 recon = recon.reshape(recon.shape[1]*recon.shape[2] , 75).detach().cpu()
#                 y = y.reshape(y.shape[1]*y.shape[2], 75).detach().cpu()

#             if (epoch + 1) % model_interval == 0:
#                 PARENT_PATH = 'model_weights'
#                 if not os.path.isdir(PARENT_PATH):
#                     os.mkdir(PARENT_PATH)
                
#                 PATH = './model_weights/model-'+str(dim_val) +'-encoder-'+ str(n_encoder_layers)+'-decoder-'+str(n_decoder_layers)+'-epochs-'+str(epoch+1)+'.pth'
#                 torch.save(model.state_dict(), PATH)
#             model.train = True

        
#     pbar.set_description('Training   Loss: '+'{:.5f}'.format(running_loss/(i+1))+ ' Saved to :'+PATH)
#     train_losses.append(running_loss/(i+1))

# # Plotting the training graph
# pyplot.figure()
# pyplot.plot(train_losses)
# if not os.path.isdir('train_graphs'):
#     os.mkdir('train_graphs')
# TRAINPATH = './train_graphs/VmRec-'+str(dim_val) +'-encoder-'+ str(n_encoder_layers)+'-decoder-'+str(n_decoder_layers)+'-epochs-'+str(epoch)+'-window_size-'+str(window_size)+'.png'
# pyplot.savefig(TRAINPATH)
# print('Training graph saved to '+ TRAINPATH)