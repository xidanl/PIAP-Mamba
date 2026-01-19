from plotters import MEA_plot,plot_apds,act_vs_pred_plot,plot_samples
from models import Mamba_model
from constants import *
import torch
import numpy as np
from utils import *
import random
import pickle
from tqdm import tqdm
from main import prepare_raw_data

def inf_intras(name, data_loader, model):
    """
    Use the specified model for iAPs prediction.

    Parameters:
    - name: train/test
    - data_loader: Ap dataset loader
    - model: specified model
    """
    print(f"Step One: start inferencing for {name}...")
    intras_list = []
    preds_list = []

    pbar = tqdm(data_loader,desc=f"Inferencing [{name}]",total=len(data_loader))
    for extras_t, intras_t, apd_t in pbar:
        extras_t = extras_t.to(device).float()  
        intras_t = intras_t.to(device).float() 
            
        with torch.no_grad():
            pred_intras, _ = model(extras_t)
    
            pred_intras = pred_intras.reshape(-1,8000).cpu().numpy()
            gt_intras = intras_t.reshape(-1,8000).cpu().numpy()

        pred_intras_new, intras_t_new = autro_correct(pred_intras, gt_intras)

        # if pred_intras_new is None or len(pred_intras_new) == 0:
        #     continue
            
        intras_list.append(intras_t_new)
        preds_list.append(pred_intras_new)
    
    intras_list_ = np.vstack(intras_list)
    preds_list_ = np.vstack(preds_list)

    sum_error_test = np.mean(np.abs(preds_list_ - intras_list_),axis=1)
    print(f"MAE:{sum_error_test.mean()}, MAE-STD:{sum_error_test.std()}, MAE-MAX:{sum_error_test.max()}")

    return intras_list_, preds_list_



if __name__ == "__main__":
    seed = 91
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_ckpt_path = "path to checkpoint"
    print(f"Start inference for step one using {model_ckpt_path}:")
    data_loc = "eap_iap_data.pkl"
    seq_len = 8000
    device = "cuda"
    
    model = Mamba_model(seq_len=8000, n_layers=4, dim=512,use_cond=False, physics=False).to(device)
    checkpoint = torch.load(model_ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    data = prepare_raw_data(data_loc)

    # train
    intras_train, extras_train = data_prep_test(data, trains,raw = False,  max_samples=10000, limit_keys=trains_with_limits)
    apd_train =  get_all_apds_multiprocessing(intras_train.reshape(-1, 8000))
    train_dataset = ApDataset(extras_train,intras_train,apd_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)

    
    train_intras, train_pred_intras = inf_intras("train",train_loader,model)
    results_data = {}
    results_data[("train","intras")] = train_intras
    results_data[("train","pred_intras")] = train_pred_intras

    with open("train_step_one_data.pkl", "wb") as f:
        pickle.dump(results_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    test_datasets = []
    test_dataloaders = []

    # test
    for test_number, test_data_name in enumerate(tests):  
        extras_test = data[(test_data_name, 'extra')]
        intras_test = data[(test_data_name, 'intra')]
        apd_test = get_all_apds_multiprocessing((data[(test_data_name, 'intra')]).reshape(-1, 8000))

        test_dataset = ApDataset(extras_test, intras_test, apd_test)
        test_datasets.append(test_dataset)

        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        test_dataloaders.append(test_dataloader)

        print(f"test dataset {test_number} {test_data_name}:{extras_test.shape}")

    test_results_data = {}
    for test_number, test_data_name in enumerate(tests): 
        test_loader = test_dataloaders[test_number]

        test_intras, test_pred_intras = inf_intras(test_data_name, test_loader,model)

        test_results_data[(f"{test_data_name}","intras")] = test_intras
        test_results_data[(f"{test_data_name}","pred_intras")] = test_pred_intras
    
    with open("test_step_one_data.pkl", "wb") as f:
        pickle.dump(test_results_data, f, protocol=pickle.HIGHEST_PROTOCOL)





