import argparse
import copy
import numpy as np
import pandas as pd
import torch
import warnings

from Data.data_sim import SimStudyNonLinearNonPH
from Data.data_sim import SimStudyNonLinearNonPHSquared
from Data.data_sim import SimStudyNonLinearNonPHCubed
from Data.data_sim import SimStudyNonLinearNonPHAll

from pycox import datasets
from pycox.evaluation import EvalSurv

from model.dataset import Dataset, sample_by_quantiles
from model.discretiser import Discretiser
from model.fedcox import Federation
from model.interpolate import surv_const_pdf_df
from model.load import read_csv
from model.net import MLP, MLPPH, CoxPH

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Seed')

def main(args):

    seed = args.seed
    rng = np.random.default_rng(seed)
    _ = torch.manual_seed(seed)

    # data = datasets.metabric.read_df()
    # data = datasets.support.read_df()
    data = datasets.gbsg.read_df()
    # data = datasets.rr_nl_nhp.read_df()

    # n = 4000
    # sims = [SimStudyNonLinearNonPH(), SimStudyNonLinearNonPHSquared(), SimStudyNonLinearNonPHCubed(), SimStudyNonLinearNonPHAll()]
    # sim = sims[0]
    # data = sim.simulate(n)
    # data = sim.dict2df(data, True)
    # data = data.drop(columns=['duration_true','event_true','censoring_true']) 

    # datapath = '../Data/data.csv'
    # data = read_csv(datapath)
    # data = data.drop(columns='PATIENTID')
    # data = data.rename(columns={'DAYS_SINCE_DIAGNOSIS':'duration', 'EVENT':'event'})

    # data = datasets.flchain.read_df()
    # data = data.rename(columns={'death' : 'event', 'futime' : 'duration'}) 

    data = data.astype({'event' : int})
    print(data.shape)

    # standardisation of features
    # simulation
    # cols_standardise = []
    # cols_minmax = ['x0', 'x1', 'x2']
    # cols_leave = []

    # metabric
    # cols_standardise = []
    # cols_minmax = ['x0', 'x1', 'x2', 'x3','x8']
    # cols_leave = ['x4','x5','x6','x7']

    # support
    # cols_standardise = []
    # cols_minmax = ['x0','x2','x3','x6','x7', 'x8', 'x9','x10','x11','x12','x13']
    # cols_leave = ['x1','x4','x5']

    # gbsg
    cols_standardise = []
    cols_minmax = ['x3', 'x4','x5', 'x6']
    cols_leave = ['x0','x1','x2']

    # flchain
    # cols_standardise = []
    # cols_minmax = ['age','sample.yr','kappa','lambda','flc.grp','creatinine']
    # cols_leave = ['mgus','sex']

    # simulacrum
    # cols_standardise = ['GRADE', 'AGE', 'QUINTILE_2015', 'NORMALISED_HEIGHT', 'NORMALISED_WEIGHT']
    # cols_minmax = ['SEX', 'TUMOUR_COUNT', 'REGIMEN_COUNT']
    # cols_leave = ['SACT', 'CLINICAL_TRIAL_INDICATOR', 'CHEMO_RADIATION_INDICATOR','BENIGN_BEHAVIOUR','SITE_C70', 'SITE_C71', 'SITE_C72', 'SITE_D32','SITE_D33','SITE_D35','CREG_L0201','CREG_L0301','CREG_L0401','CREG_L0801','CREG_L0901','CREG_L1001','CREG_L1201','CREG_L1701','LAT_9','LAT_B','LAT_L','LAT_M','LAT_R','ETH_A','ETH_B','ETH_C','ETH_M','ETH_O','ETH_U','ETH_W','DAYS_TO_FIRST_SURGERY']

    all_cols = cols_standardise + cols_minmax + cols_leave
    standardise = [(f'standard{i}',StandardScaler(), [col]) for i,col in enumerate(cols_standardise)]
    leave = [(f'leave{i}','passthrough',[col]) for i,col in enumerate(cols_leave)]
    minmax = [(f'minmax{i}',MinMaxScaler(),[col]) for i,col in enumerate(cols_minmax)] 

    x_mapper = ColumnTransformer(standardise + minmax + leave) 

    # discretisation
    num_durations = 40
    discretiser = Discretiser(num_durations, scheme='km')

    """
    Argument:
    x - DataFrame of features
    y - tuple of (durations, events)
    x_mapper - ColumnTransformer for all features
    discretiser - Discretiser to be applied to y
    fit_transform - for x_mapper and discretiser on x and y respectively 
    Returns:
    x_trans - 
    y_trans - tuple of (discretised durations, events)
    """
    def data_transform(x, y, x_mapper, discretiser, fit_transform=True):

        if fit_transform:
            x_trans = x_mapper.fit_transform(x).astype('float32')
        else:
            x_trans = x_mapper.transform(x).astype('float32')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if fit_transform:
                y_trans = discretiser.fit_transform(*y)
            else:
                y_trans = discretiser.transform(*y)   

        return x_trans, y_trans

    """
    Argument:
    df - simple DataFrame with all features and labels 
    t_index - rows to be assigned to train
    v_index - rows to be assigned to val
    features_headers - list of feature names
    Returns:
    x_train, x_val - df projection containing only features 
    y_train, y_val- tuple of (durations, events)
    """
    def train_val_split(df, t_index, v_index, feature_headers):
        df_train = df.loc[t_index]
        df_val = df.loc[v_index]

        x_train = df_train[feature_headers]
        y_train = (df_train.duration.values, df_train.event.values)
        x_val = df_val[feature_headers]
        y_val = (df_val.duration.values, df_val.event.values)

        return x_train, y_train, x_val, y_val


    case1 = {'num_centers' : 1, 
                'local_epochs' : [1],
                'stratify_labels' : False,
                'case_id' : 'central'}

    case2 = {'num_centers' : 4, 
                'local_epochs' : [1,5,20,100],
                'stratify_labels' : False,
                'case_id' : 'iid'}

    case3 = {'num_centers' : 4, 
                'local_epochs' : [1,5,20,100],
                'stratify_labels' : True,
                'case_id' : 'noniid'}

    ### just for val losses
    # if True:
    #     case2 = {'num_centers' : 4, 
    #                 'local_epochs' : [1],
    #                 'stratify_labels' : False,
    #                 'case_id' : 'iid'}

    #     case3 = {'num_centers' : 4, 
    #                 'local_epochs' : [1],
    #                 'stratify_labels' : True,
    #                 'case_id' : 'noniid'}


    cases = [case1, case2, case3]
    # cases = [case3]
    
    model_type = 'CoxPH'
    loss_folder = f'../results-gbsg-40/losses'
    log_folder = f'../results-gbsg-40/{model_type}'
    test_by_center = True

    # set at high number to deactivate
    reset_in = 60000 

    for case in cases:

        # if equal to 1 only once and only on the first fold of case 1, if equal to ev-folds times para-folds then every time in case 1
        tune_tries = 5
        para_round = 0

        best_lr = 0.1
        best_dropout = 0
        tuning = True

        reset_in = reset_in - 1
        if reset_in == 0:
            rng = np.random.default_rng(seed)
            _ = torch.manual_seed(seed)  
            reset_in = 6      

        case_id = case['case_id']
        
        # federation parameters - excl lr
        num_centers = case['num_centers']
        optimizer = 'adam'
        batch_size = 256
        local_epochs = 1 # overridden below
        base_epochs = 100
        print_every = 100
        # no stratification if None and False
        stratify_col = None
        stratify_labels = case['stratify_labels']

        # this is set automatically
        stratify_on = None

        if stratify_col != None:
            stratify_on = all_cols.index(stratify_col)
            print(f'Stratify on index: {stratify_on}')
        if stratify_labels:
            stratify_on = 0
            print(f'Stratify on label index: {stratify_on}')
            
        # case level
        for local_epochs in case['local_epochs']:
            
            epochs = max(1, base_epochs // local_epochs)

            log = f'{log_folder}/training_log_M{model_type}C{case_id}S{stratify_on}C{num_centers}L{local_epochs}.txt'
            with open(log, 'w') as f:
                print(f'-- Centers: {num_centers}, Local rounds: {local_epochs}, Global rounds: {epochs} --', file=f)

            case_local_val_losses = []
            case_global_val_losses = []
            case_local_train_losses = []
            case_global_train_losses = []

            # CV setup
            n_splits = 5
            random_state = rng.integers(0,1000)
            scores = []
            briers = []
            parameters = []

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            cv_round = 0
            
            # CV for average performance
            for train_index, test_index in kf.split(data):
                with open(log, 'a') as f:
                    print(f'-- Eval CV fold: {cv_round} --', file=f)
                cv_round += 1
                x_train, y_train, x_test, y_test = train_val_split(data, train_index, test_index, all_cols)
                x_train_trans, y_train_trans = data_transform(x_train, y_train, x_mapper, discretiser, fit_transform=True)
                x_test_trans, _ = data_transform(x_test, y_test, x_mapper, discretiser, fit_transform=False) # leaving y_test undiscretised

                test_loader = DataLoader(Dataset(x_test_trans, y_test), batch_size=256, shuffle=False)

                # MLP parameters - excl dropout
                dim_in = x_train_trans.shape[1]
                num_nodes = [32, 32]
                dim_out = len(discretiser.cuts)
                batch_norm = True

                # tuning
                if tuning:
                    # grid for parameter 1 to be tuned
                    learning_rates = [0.1, 0.01, 0.001, 0.0001]

                    # grid for parameter 2 to be tuned
                    # >>> for CoxPH just set to 0 - doesn't make a difference
                    # dropouts = [0.1, 0.5, 0.75] 
                    dropouts = [0]
                    
                    # [[scores for each lr x dropout from fold 1], [..from fold2], etc.]
                    tuning_scores = []    

                    para_splits = 5
                    para_kf = KFold(n_splits=para_splits)
                    for t_index, v_index in kf.split(x_train_trans):
                        
                        x_t, y_t, x_v, y_v = train_val_split(data.loc[train_index].reset_index(), t_index, v_index, all_cols)
                        x_t_trans, y_t_trans = data_transform(x_t, y_t, x_mapper, discretiser, fit_transform=False)
                        x_v_trans, _ = data_transform(x_v, y_v, x_mapper, discretiser, fit_transform=False) # leaving y_v undiscretised

                        val_loader = DataLoader(Dataset(x_v_trans, y_v), batch_size=256, shuffle=False)

                        # each entry corresponds to the score for a particular lr x dropout pair
                        fold_scores = []
                        for lr in learning_rates:
                            for dropout in dropouts:
                                
                                para_epochs = max(1, epochs // 5)

                                if model_type == 'NNnph':   
                                    net = MLP(dim_in=dim_in, num_nodes=num_nodes, dim_out=dim_out, batch_norm=batch_norm, dropout=dropout)
                                if model_type == 'CoxPH':
                                    net = CoxPH(dim_in=dim_in, dim_out=dim_out, batch_norm=batch_norm)
                                if model_type == 'NNph':
                                    net = MLPPH(dim_in=dim_in, num_nodes=num_nodes, dim_out=dim_out, batch_norm=batch_norm, dropout=dropout)
                                else:
                                    ValueError

                                fed = Federation(features=x_t_trans, labels=y_t_trans, net=net, num_centers=num_centers, optimizer=optimizer, lr=lr, stratify_on=stratify_on, stratify_labels=stratify_labels, batch_size=batch_size, local_epochs=local_epochs, raw_labels=y_t)
                                ran_for = fed.fit(epochs=para_epochs, patience=999, print_every=print_every, take_best=True, verbose=False)    

                                surv = fed.predict_surv(val_loader)[0]
                                surv = surv_const_pdf_df(surv, discretiser.cuts) # interpolation

                                ev = EvalSurv(surv, *y_v, censor_surv='km')
                                score = ev.concordance_td('antolini')
                                fold_scores.append(score)
                                with open(log, 'a') as f:
                                    print(f'Tuning CV fold {para_round} with {ran_for} rounds: conc = {score}, lr = {lr}, dropout = {dropout}', file=f)
                        tuning_scores.append(fold_scores)
                        
                        para_round += 1
                        if para_round >= tune_tries:
                            tuning = False
                            break # out of para loop

                    tuning_scores = np.array(tuning_scores)
                    avg_scores = np.mean(tuning_scores, axis=0)
                    best_combo_idx = np.argmax(avg_scores)
                    best_lr_idx = best_combo_idx // len(dropouts)
                    best_dropout_idx = best_combo_idx % len(dropouts)
                    best_lr = learning_rates[best_lr_idx]
                    best_dropout = dropouts[best_dropout_idx]

                if model_type == 'NNnph':   
                    net = MLP(dim_in=dim_in, num_nodes=num_nodes, dim_out=dim_out, batch_norm=batch_norm, dropout=best_dropout)    
                if model_type == 'CoxPH':            
                    net = CoxPH(dim_in=dim_in, dim_out=dim_out, batch_norm=batch_norm)
                if model_type == 'NNph':
                    net = MLPPH(dim_in=dim_in, num_nodes=num_nodes, dim_out=dim_out, batch_norm=batch_norm, dropout=best_dropout)
                else:
                    ValueError

                fed = Federation(features=x_train_trans, labels=y_train_trans, net=net, num_centers=num_centers, optimizer=optimizer, lr=best_lr, stratify_on=stratify_on, stratify_labels=stratify_labels, batch_size=batch_size, local_epochs=local_epochs, raw_labels=y_train)
                ran_for = fed.fit(epochs=epochs, patience=999, print_every=print_every, take_best=True)    
                
                surv = fed.predict_surv(test_loader)[0]
                surv = surv_const_pdf_df(surv, discretiser.cuts) # interpolation
                
                time_grid = np.linspace(y_test[0].min(), y_test[0].max(), 100)
                
                if test_by_center:
                    dict_center_idxs_test = sample_by_quantiles(y_test,0,4)
                    for center in dict_center_idxs_test:
                        idxs_test = dict_center_idxs_test[center]
                        ev = EvalSurv(surv.iloc[:, idxs_test], y_test[0][idxs_test], y_test[1][idxs_test], censor_surv='km')
                        score = ev.concordance_td('antolini')
                        brier = ev.integrated_brier_score(time_grid) 
                        with open(log, 'a') as f:
                            print(f'>> Center {center}: conc = {score}, brier = {brier}, LR = {best_lr}, dropout = {best_dropout}', file=f)
                ev = EvalSurv(surv, *y_test, censor_surv='km')
                score = ev.concordance_td('antolini')
                scores.append(score)

                brier = ev.integrated_brier_score(time_grid) 
                briers.append(brier)
                with open(log, 'a') as f:
                    print(f'>> After {ran_for} rounds, model from round {fed.model_from_round}: conc = {score}, brier = {brier}, LR = {best_lr}, dropout = {best_dropout}', file=f)

                parameters.append({'lr' : best_lr, 'dropout' : best_dropout})
                case_local_val_losses.append(fed.local_val_losses)
                case_global_val_losses.append(fed.global_val_losses)
                case_local_train_losses.append(fed.local_train_losses)
                case_global_train_losses.append(fed.global_train_losses)


            losses = np.array(case_local_val_losses)
            lossfile = f'{loss_folder}/local_val_loss_M{model_type}C{case_id}L{local_epochs}.npy'
            np.save(lossfile, losses)

            losses = np.array(case_global_val_losses)
            lossfile = f'{loss_folder}/global_val_loss_M{model_type}C{case_id}L{local_epochs}.npy'
            np.save(lossfile, losses)

            losses = np.array(case_local_train_losses)
            lossfile = f'{loss_folder}/local_train_loss_M{model_type}C{case_id}L{local_epochs}.npy'
            np.save(lossfile, losses)

            losses = np.array(case_global_train_losses)
            lossfile = f'{loss_folder}/global_train_loss_M{model_type}C{case_id}L{local_epochs}.npy'
            np.save(lossfile, losses)

            with open(log, 'a') as f:
                print(f'Avg concordance: {sum(scores) / len(scores)}, Integrated Brier: {sum(briers) / len(briers)}', file=f)






if __name__ == "__main__":
    args = parse_args()
    main(args)