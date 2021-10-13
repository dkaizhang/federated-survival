import numpy as np

def get_filepath(dura):
    path1 = f'../{dura}/coxph-{dura}'
    path2 = f'../{dura}/nnph-{dura}'
    path3 = f'../{dura}/nnnph-{dura}'
    paths = [path1, path2, path3]
    model = ['coxph', 'NNph', 'NNnph']
    files = []
    for i, path in enumerate(paths):
        cases = ['central','iid','noniid']
        strats = [None, None, 0]
        centers = [1,4,4]
        rounds = [[1], [1,5,20,100], [1,5,20,100]]
        for j, case in enumerate(cases):  
            for rnd in rounds[j]:
                file = f'{path}/training_log_M{model[i]}C{case}S{strats[j]}C{centers[j]}L{rnd}.txt'
                files.append(file)
    return files

def get_filepath_additional(dura):
    path1 = f'../{dura}/CoxPH'
    path2 = f'../{dura}/NNph'
    path3 = f'../{dura}/NNnph'
    paths = [path1, path2, path3]
    model = ['CoxPH', 'NNph', 'NNnph']
    files = []
    for i, path in enumerate(paths):
        cases = ['central','iid','noniid']
        strats = [None, None, 0]
        centers = [1,4,4]
        rounds = [[1], [1,5,20,100], [1,5,20,100]]
        for j, case in enumerate(cases):  
            for rnd in rounds[j]:
                file = f'{path}/training_log_M{model[i]}C{case}S{strats[j]}C{centers[j]}L{rnd}.txt'
                files.append(file)
    return files

def extract_figures(files):
    concordances = []
    briers = []
    for file in files:
        with open(file, 'r') as f:
            # print(file)
            lines = f.read().splitlines()
            last_line = lines[-1]
            # print(last_line)
            start = last_line.find('concordance: ') + len('concordance: ') 
            end = 10        
            concordance = float(last_line[start:start+end])
            concordances.append(concordance)
            # print(round(concordance * 100, 2))
            start = last_line.find('Brier: ') + len('Brier: ') 
            end = 10     
            brier = float(last_line[start:start+end])
            briers.append(brier)
            # print(round(brier * 100, 2))
    return concordances, briers

def extract_stats(files):
    indiv_rounds = []
    avg_rounds = []
    std_rounds = []
    indiv_concordances = []
    avg_concordances = []
    std_concordances = []
    indiv_briers = []
    avg_briers = []
    std_briers = []
    for file in files:
        with open(file, 'r') as f:
            # print(file)
            lines = f.read().splitlines()
            # line_pos = [2,4,6,8,10]
            line_pos = [2,8,14,20,26]
            concs = []
            brs = []
            rounds = []
            for lp in line_pos:
                line = lines[-lp] 
                start = line.find('conc = ') + len('conc = ')
                end = 10
                conc = float(line[start:start+end])
                concs.append(conc)
                start = line.find('brier = ') + len('brier = ')
                br = float(line[start:start+end])
                brs.append(br)
                start = line.find('from round ') + len('from round ') 
                end = 2     
                if line[start:start+end][-1] == ':':
                    rnd = int(line[start:start+end-1])
                else:
                    rnd = int(line[start:start+end])
                rounds.append(rnd)
            indiv_rounds.append(rounds)
            avg_rounds.append(np.mean(rounds))
            std_rounds.append(np.std(rounds))
            indiv_concordances.append(concs)
            avg_concordances.append(np.mean(concs))
            std_concordances.append(np.std(concs))
            indiv_briers.append(brs)
            avg_briers.append(np.mean(brs))
            std_briers.append(np.std(brs))
    return indiv_rounds, avg_rounds, std_rounds, indiv_concordances, avg_concordances, std_concordances, indiv_briers, avg_briers, std_briers

def extract_lr(files):
    lrs = []

    for file in files:
        with open(file, 'r') as f:
            # print(file)
            lines = f.read().splitlines()
            line_pos = [2]

            for lp in line_pos:
                line = lines[-lp] 
                start = line.find('LR = ') + len('LR = ')
                end = line.find(', dropout')
                lr = float(line[start:end])
                lrs.append(lr)
    return lrs