import pandas as pd

def predict_hazard(model, data, device):
    model.to(device)
    model.eval()

    x, durations, events = data
    x = x.to(device)
    durations = durations.to(device)
    events = events.to(device)

    hazard = model(x).sigmoid()   
    model.train()
    return hazard

def predict_surv(model, data, device):
    hazard = predict_hazard(model, data, device) # size batch x num_durations
    surv = (1 - hazard).log().cumsum(1).exp() # size batch x num_durations
    return surv.cpu().detach()

def predict_surv_df(model, data, cuts, device):
    surv = predict_surv(model, data, device)
    return pd.DataFrame(surv.transpose(), cuts)