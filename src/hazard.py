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

def predict_surv(model, data):
    hazard = predict_hazard(model, data)
    surv = (1 - hazard).log().cumsum(1).exp()
    return surv.cpu().detach().numpy()

def predict_surv_df(model, data, cuts):
    surv = predict_surv(model, data)
    return pd.DataFrame(surv.transpose(), cuts)