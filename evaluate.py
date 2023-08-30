
import numpy as np
import torch
import yaml

from argparse import ArgumentParser
from pycox.evaluation import EvalSurv
from src.data import load_data, load_raw_data, get_min_max_durations
from src.discretiser import Discretiser
from src.hazard import predict_surv
from src.interpolate import surv_const_pdf_df
from src.net import load_model
from torch.utils.data import DataLoader

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    return parser.parse_args()

def main(args):

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    discretiser = Discretiser(config['num_durations'], scheme='km')

    _, _, test = load_data(dataset=config['dataset'], num_durations=config['num_durations'], seed=config['seed'])
    _, _, raw_test = load_raw_data(dataset=config['dataset'], seed=config['seed'])

    loader = DataLoader(test, batch_size=config['batch_size'], shuffle=False)

    model = load_model(model_type=config['model_type'], dim_in=test[0][0].shape[0], dim_out=config['num_durations'],model_path=config['eval_path'])

    device = f"cuda:{config['device']}" if torch.cuda.is_available() else "cpu"

    surv = []
    for data in loader:
        surv.append(predict_surv(model, data, device)) # TODO why [0] in predict_surv(test_loader)[0]
    surv = torch.cat(surv, dim=0)

    surv = surv_const_pdf_df(surv, discretiser.cuts) # interpolation
    
    min_duration, max_duration = get_min_max_durations(dataset=config['dataset'], split='test', seed=config['seed'])

    time_grid = np.linspace(min_duration, max_duration, 100)
    
    # if test_by_center:
    #     dict_center_idxs_test = sample_by_quantiles(y_test,0,4)
    #     for center in dict_center_idxs_test:
    #         idxs_test = dict_center_idxs_test[center]
    #         ev = EvalSurv(surv.iloc[:, idxs_test], y_test[0][idxs_test], y_test[1][idxs_test], censor_surv='km')
    #         score = ev.concordance_td('antolini')
    #         brier = ev.integrated_brier_score(time_grid) 
    #         with open(log, 'a') as f:
    #             print(f'>> Center {center}: conc = {score}, brier = {brier}, LR = {best_lr}, dropout = {best_dropout}', file=f)
    ev = EvalSurv(surv, raw_test.duration.to_numpy(), raw_test.event.to_numpy(), censor_surv='km')
    score = ev.concordance_td('antolini')

    brier = ev.integrated_brier_score(time_grid) 

    print("Concordance", score)
    print("brier", brier)

if __name__ == '__main__':
    args = parse_args()
    main(args)