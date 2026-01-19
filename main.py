import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from torch import optim
from utils import *
from plotters import MAE_plot,plot_apds,act_vs_pred_plot_with_residual
import logging
from torch.utils.tensorboard import SummaryWriter
import warnings
from models import Mamba_model
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

# seed everything
seed=91
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def prepare_raw_data(data_loc="eap_iap_data.pkl"):
    """
    Electrophysiological Data Preprocessing Module

    Processes extracellular (eAP) and intracellular (iAP) action potential recordings from pickle files. 
    Filters high-quality signal pairs and applies normalization procedures for downstream analysis.
    """
    
    with open(data_loc, 'rb') as inp:
        all_recordings = pickle.load(inp)
    data={}
    all_data_keys = all_recordings.keys()
    for key in all_data_keys:    
        sn,intras,extras,intras_normalized,extras_normalized = all_recordings[key]
        mins = extras_normalized[:, 4000:5000].min(axis=1) > -0.3
        sn,intras,extras,intras_normalized,extras_normalized = sn[mins],intras[mins],extras[mins],intras_normalized[mins],extras_normalized[mins]

        intras_normalized2 = intras_normalized+0.11
        intras_normalized2 = intras_normalized2/1.1
        extras_normalized2 = extras_normalized - extras_normalized[:, 0][:, np.newaxis]
    
        if len (intras) > 10:
            data[(key,'extra_norm1')] = extras_normalized[10:-5]
            data[(key,'extra')]=extras_normalized2[10:-5]
            data[(key,'intra')]=intras_normalized2[10:-5]
            data[(key,'extra_raw')]=extras[10:-5]
            data[(key,'intra_raw')]=intras[10:-5]
            data[(key,'s/n')]=sn[10:-5]

    del all_recordings

    return data

class PI2MambaTrainer():
    """
    A wrapper for the whole training and evaluating process.

    Parameters:
    - step: contorlling stage one or stage two
    - train_data_loc / test_data_loc: path to your data
    - args: other hyperparameters
    """
    def __init__(self, step, train_data_loc, test_data_loc, args):

        self.train_loader = None
        self.test_dataloaders = []
        self.args = args

        print(f"Setting up for step {step} ...")

        if step == 'one':

            # Dataset
            data = prepare_raw_data(train_data_loc)
        
            intras_train, extras_train = data_prep_test(data, trains,raw = False,  max_samples=args.max_samples, limit_keys=trains_with_limits)
            apd_train =  get_all_apds_multiprocessing(intras_train.reshape(-1, 8000))
            train_dataset = ApDataset(extras_train,intras_train,apd_train)
            self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            print(f"train dataset:{extras_train.shape}")

            for test_number, test_data_name in enumerate(tests):  
                extras_test = data[(test_data_name, 'extra')]
                intras_test = data[(test_data_name, 'intra')]
                apd_test = get_all_apds_multiprocessing((data[(test_data_name, 'intra')]).reshape(-1, 8000))

                test_dataset = ApDataset(extras_test, intras_test, apd_test)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
                self.test_dataloaders.append(test_dataloader)

                print(f"test dataset {test_number} {test_data_name}:{extras_test.shape}")
        
        elif step == "two":
            # Dataset
            with open(train_data_loc, 'rb') as inp:
                data = pickle.load(inp)

            pred_intras_train = data[("train", "pred_intras")]
            intras_train = data[("train","intras")]
            apd_train = get_all_apds_multiprocessing(intras_train.reshape(-1, 8000))

            train_dataset = ApDataset(pred_intras_train, intras_train, apd_train)
            self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            print(f"train dataset:{intras_train.shape}")

            with open(test_data_loc,'rb') as t_inp:
                t_data = pickle.load(t_inp)
            
            for test_number, test_data_name in enumerate(tests):  
                extras_test = t_data[(test_data_name, 'pred_intras')]
                intras_test = t_data[(test_data_name, 'intras')]
                apd_test = get_all_apds_multiprocessing((t_data[(test_data_name, 'intras')]).reshape(-1, 8000))

                test_dataset = ApDataset(extras_test, intras_test, apd_test)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
                self.test_dataloaders.append(test_dataloader)

                print(f"test dataset {test_number} {test_data_name}:{extras_test.shape}")

        print("============data loaded===============")

        # model
        self.model = Mamba_model(seq_len=args.seq_len, n_layers=args.n_layers, dim=512, use_cond=args.use_cond, physics=args.physics).to(args.device)
        print(f"total params: {sum(p.numel() for p in self.model.parameters())/ 1000000:.2f}M")

        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name.lower() or "bias" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": 0.001},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=args.lr
        )
        self.clip_value = 2.0
        self.mae = nn.L1Loss()
    
    def train(self):
        setup_logging(self.args.run_name)
        logger = SummaryWriter(os.path.join("runs",self.args.run_name))
        l = len(self.train_loader)

        for epoch in range(1,self.args.epochs+1):
            self.model.train()
            logging.info(f"Starting epoch {epoch}/{self.args.epochs}:")

            pbar = tqdm(self.train_loader)
            epoch_loss = 0.0
            for i, (extras, intras, _) in enumerate(pbar):
                extras = extras.to(self.args.device).float()    
                intras = intras.to(self.args.device).float()

                pred_intras, pred_dv_out = self.model(extras)
        
                intras_loss = self.mae(pred_intras,intras)
                if self.args.physics:
                    dv_out_groud_truth = torch.zeros(pred_intras.shape[0], 7700, device=pred_intras.device)
                    physics_loss = self.mae(pred_dv_out,dv_out_groud_truth)
                    loss = ( intras_loss * self.args.w_data + physics_loss * self.args.w_physics ) / (self.args.w_data + self.args.w_physics)
                else:
                    loss = intras_loss

                self.optimizer.zero_grad()
                loss.backward()

                if self.clip_value > 0:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.clip_value
                    )
                    logger.add_scalar("Gradient_Norm", total_norm, global_step=epoch * l + i)

                self.optimizer.step()

                if self.args.physics:
                    pbar.set_postfix({
                        'MAE(weighted)':loss.item(),
                        'data':intras_loss.item(),
                        'physics':physics_loss.item()
                    })
                else:
                    pbar.set_postfix(MAE=loss.item())
                
                logger.add_scalar("MAE_batch", loss.item(), global_step=epoch * l + i)
                epoch_loss += loss.item()
            
            epoch_avg_loss = epoch_loss / len(self.train_loader)
            logger.add_scalar("MAE_epoch", epoch_avg_loss, global_step=epoch)
            
            if epoch >= 50 and epoch % 5 == 0:
                # saving model
                torch.save(self.model.state_dict(), os.path.join("models", self.args.run_name, f"ckpt_{epoch}.pt"))
                torch.save(self.optimizer.state_dict(), os.path.join("models", self.args.run_name, f"optim_{epoch}.pt"))

                print(f"Starting evaluate for epoch {epoch}")
                self.evaluate(save_pic=True)
    
    def evaluate(self, model_ckpt_path=None, save_pic=False, combine=False):
        
        if model_ckpt_path:
            print(f"Starting test for {model_ckpt_path}:")
            model = Mamba_model(seq_len=8000, n_layers=4, dim=512,use_cond=False, physics=True).to(self.args.device)
            checkpoint = torch.load(model_ckpt_path, map_location=self.args.device, weights_only=True)
            model.load_state_dict(checkpoint, strict=True)
        else:
            model = self.model
        
        model.eval()

        intras_list_all = []
        preds_list_all = []
        preds_apd_list_all = []
        actual_apds_list = []

        for test_number, test_data_name in enumerate(tests): 
            
            test_loader = self.test_dataloaders[test_number]
                    
            intras_list = []
            apds_list = []
            preds_list = []
            preds_apd_list = []

            pbar = tqdm(test_loader,desc=f"Evaluating [{test_data_name}]",total=len(test_loader))
            for extras_t, intras_t, apd_t in pbar:     
                extras_t = extras_t.to(self.args.device).float()  
                intras_t = intras_t.to(self.args.device).float() 
                
                with torch.no_grad():
                    pred_intras, _ = model(extras_t)

                    if combine:
                        pred_intras = (0.2) * pred_intras + (0.8) * extras_t 
        
                    pred_intras = pred_intras.reshape(-1,8000).cpu().numpy()
                    gt_intras = intras_t.reshape(-1,8000).cpu().numpy()

                pred_intras_new, intras_t_new = autro_correct(pred_intras, gt_intras)

                if pred_intras_new is None or len(pred_intras_new) == 0:
                    continue

                pred_intras_apd = get_all_apds_multiprocessing(pred_intras_new)
                        
                intras_list.append(intras_t_new)
                apds_list.append(apd_t)
                preds_list.append(pred_intras_new)
                preds_apd_list.append(pred_intras_apd)
                    
            intras_list_ = np.vstack(intras_list)
            apds_list_ = np.vstack(apds_list)
            preds_list_ = np.vstack(preds_list)
            preds_apd_list_ = np.vstack(preds_apd_list)

            intras_list_all.append(intras_list_)
            actual_apds_list.append(apds_list_)
            preds_list_all.append(preds_list_)
            preds_apd_list_all.append(preds_apd_list_)

            sum_error_test = np.mean(np.abs(preds_list_ - intras_list_),axis=1)

            print(f"Test dataset {test_data_name} finished.")
            print(f"MAE:{sum_error_test.mean():.4f}, MAE-STD:{sum_error_test.std():.4f}, MAE-MAX:{sum_error_test.max():.4f}")

            mse_per_sample = np.mean((preds_list_ - intras_list_) ** 2, axis=1)
            print(f"MSE:{mse_per_sample.mean():.4f}, MSE-STD:{mse_per_sample.std():.4f}, MSE-MAX:{mse_per_sample.max():.4f}")

            if save_pic:
                act_vs_pred_plot_with_residual(intras_list_, preds_list_,label=test_data_name,color='blue',plot_name=f"test{test_number}_scatter")

        if save_pic:    
            MAE_plot("model",preds_list_all, intras_list_all)
            plot_apds("model", actual_apds_list, preds_apd_list_all)
            

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "Training"
    args.epochs = 120
    args.n_layers = 2
    args.max_samples=10000
    args.batch_size = 32
    args.train = True
    args.physics = True
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.seq_len = 8000
    args.lr = 2e-4
    args.use_cond = False
    args.w_data = 10
    args.w_physics = 0.05
    
    trainer = PI2MambaTrainer(
        step='two', # switching between 'one' and 'two'
        train_data_loc="path to training data",
        test_data_loc="path to test data",
        args=args
    )

    # train
    trainer.train()

    # eval: uncomment the belowing lines to start evaluate for your trained model
    # model_ckpt_path = "path to checkpoint"
    # trainer.evaluate(model_ckpt_path=model_ckpt_path, save_pic=True)
    


    


    