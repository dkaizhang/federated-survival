import os
import pandas as pd
import socket
import torch

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def get_summarywriter(out_dir):

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(
        out_dir, current_time + "_" + socket.gethostname()
    )
    return SummaryWriter(log_dir=log_dir)

def predict_hazard(model, input=None):
    model.to(device)
    model.eval()
    loader = input if not None else self.testloader 
    hazard = torch.cat([model(data[0].to(device)).sigmoid() for data in loader], axis=0)         
    model.train()
    return hazard

def predict_surv(model, input=None):
    hazard = predict_hazard(model, input)
    surv = (1 - hazard).log().cumsum(1).exp()
    return surv.cpu().detach().numpy()

def predict_surv_df(model, cuts, input=None):
    surv = predict_surv(model, input)
    return pd.DataFrame(surv.transpose(), cuts)