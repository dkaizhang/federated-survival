# this provides some saving and loading utilities for csv files

import os
import pandas as pd

table_names = [
    'av_patient',
    'av_tumour',
    'sact_cycle',
    'sact_drug_detail',
    'sact_outcome',
    'sact_patient',
    'sact_regimen',
    'sact_tumour'
]

def load_table(table_name,
               path):
    """
    Loads specified table and returns it.
    For the standard tables loads them with recommended types for each column.
    """
    prefix = 'sim_'

    default_dtypes = {
        'av_patient' : {
            'PATIENTID' : int,
            'SEX' : 'category',
            'LINKNUMBER' : int,
            'ETHNICITY' : 'category',
            'DEATHCAUSECODE_1A' : object,
            'DEATHCAUSECODE_1B' : object,
            'DEATHCAUSECODE_1C' : object,
            'DEATHCAUSECODE_2' : object,
            'DEATHCAUSECODE_UNDERLYING' : object,
            'DEATHLOCATIONCODE' : 'category',
            'NEWVITALSTATUS' : 'category',
            'VITALSTATUSDATE': object
        },
        'av_tumour' : {
            'TUMOURID' : int,
            'PATIENTID' : int,
            'DIAGNOSISDATEBEST' : object,
            'SITE_ICD10_O2' : 'category',
            'SITE_ICD10_O2_3CHAR' : 'category',
            'MORPH_ICD10_O2' : 'category',
            'BEHAVIOUR_ICD10_O2' : 'category',
            'T_BEST' : 'category',
            'N_BEST' : 'category',
            'M_BEST' : 'category',
            'STAGE_BEST' : 'category',
            'STAGE_BEST_SYSTEM' : 'category',
            'GRADE' : 'category',
            'AGE' : float,
            'SEX' : 'category',
            'CREG_CODE' : 'category',
            'LINK_NUMBER' : int,
            'SCREENINGSTATUSFULL_CODE' : 'category',
            'ER_STATUS' : 'category',
            'ER_SCORE' : 'category',
            'PR_STATUS' : 'category',
            'PR_SCORE' : 'category',
            'HER2_STATUS' : 'category',
            'CANCERCAREPLANINTENT' : 'category',
            'PERFORMANCESTATUS' : 'category',
            'CNS' : 'category',
            'ACE27' : 'category',
            'GLEASON_PRIMARY' : 'category',
            'GLEASON_SECONDARY' : 'category',
            'GLEASON_TERTIARY' : 'category',
            'GLEASON_COMBINED' : 'category',
            'DATE_FIRST_SURGERY' : object,
            'LATERALITY' : 'category',
            'QUINTILE_2015' : 'category'
        },
        'sact_cycle' : {
            'MERGED_CYCLE_ID' : int,
            'MERGED_REGIMEN_ID' : int,
            'CYCLE_NUMBER' : int,
            'START_DATE_OF_CYCLE' : object,
            'OPCS_PROCUREMENT_CODE' : 'category',
            'PERF_STATUS_START_OF_CYCLE' : 'category',
            'MERGED_PATIENT_ID' : int,
            'MERGED_TUMOUR_ID' : int
        },
        'sact_drug_detail' : {
            'MERGED_DRUG_DETAIL_ID' : int,
            'MERGED_CYCLE_ID' : int,
            'ORG_CODE_OF_DRUG_PROVIDER' : 'category',
            'ACTUAL_DOSE_PER_ADMINISTRATION' : float,
            'OPCS_DELIVERY_CODE' : 'category',
            'ADMINISTRATION_ROUTE' : 'category',
            'ADMINISTRATION_DATE' : object,
            'DRUG_GROUP' : 'category',
            'MERGED_PATIENT_ID' : int,
            'MERGED_TUMOUR_ID' : int,
            'MERGED_REGIMEN_ID' : int
        },
        'sact_outcome' : {
            'MERGED_OUTCOME_ID' : int,
            'MERGED_REGIMEN_ID' : int,
            'DATE_OF_FINAL_TREATMENT' : object,
            'REGIMEN_MOD_DOSE_REDUCTION' : 'category',
            'REGIMEN_MOD_TIME_DELAY' : 'category',
            'REGIMEN_MOD_STOPPED_EARLY' : 'category',
            'REGIMEN_OUTCOME_SUMMARY' : 'category',
            'MERGED_PATIENT_ID' : int,
            'MERGED_TUMOUR_ID' : int
        },
        'sact_patient' : {
            'MERGED_PATIENT_ID' : int,
            'LINK_NUMBER' : int
        },
        'sact_regimen' : {
            'MERGED_REGIMEN_ID' : int,
            'MERGED_TUMOUR_ID' : int,
            'HEIGHT_AT_START_OF_REGIMEN' : float,
            'WEIGHT_AT_START_OF_REGIMEN' : float,
            'INTENT_OF_TREATMENT' : 'category',
            'DATE_DECISION_TO_TREAT' : object,
            'START_DATE_OF_REGIMEN' : object,
            'MAPPED_REGIMEN' : object,
            'CLINICAL_TRIAL' : 'category',
            'CHEMO_RADIATION' : 'category',
            'MERGED_PATIENT_ID' : int,
            'BENCHMARK_GROUP' : 'category'
        },
        'sact_tumour' : {
            'MERGED_TUMOUR_ID' : int,
            'MERGED_PATIENT_ID' : int,
            'CONSULTANT_SPECIALITY_CODE' : 'category',
            'PRIMARY_DIAGNOSIS' : 'category',
            'MORPHOLOGY_CLEAN' : 'category'
        }
    }
    default_dates = {
        'av_patient' : ['VITALSTATUSDATE'],
        'av_tumour' : ['DIAGNOSISDATEBEST', 'DATE_FIRST_SURGERY'],
        'sact_cycle' : ['START_DATE_OF_CYCLE'],
        'sact_drug_detail' : ['ADMINISTRATION_DATE'],
        'sact_outcome' : ['DATE_OF_FINAL_TREATMENT'],
        'sact_patient' : [],
        'sact_regimen' : ['DATE_DECISION_TO_TREAT', 'START_DATE_OF_REGIMEN'],
        'sact_tumour' : []
    }

    # set to defaults
    table_name = table_name.lower()
    if table_name in table_names:
        dtype=default_dtypes[table_name]
        parse_dates=default_dates[table_name]

    read_path = os.path.join(path, prefix + table_name.lower() + ".csv")
    try:
        if table_name == 'sact_regimen':
            table = pd.read_csv(read_path, quotechar='"', dtype=dtype, parse_dates=parse_dates,encoding="ISO-8859-1")
        else:
            table = pd.read_csv(read_path, quotechar='"', dtype=dtype, parse_dates=parse_dates)
    except FileNotFoundError:
        raise ValueError("The file " + read_path + " does not exist.")

    return table

def to_csv(df, path, index=False):
    # Prepend dtypes to the top of df
    df2 = df.copy()
    df2.loc[-1] = df2.dtypes
    df2.index = df2.index + 1
    df2.sort_index(inplace=True)
    # Then save it to a csv
    df2.to_csv(path, index=index)

def read_csv(path):
    # Read types first line of csv
    dtypes = {key:value for (key,value) in pd.read_csv(path,    
              nrows=1).iloc[0].to_dict().items() if 'date' not in value}

    parse_dates = [key for (key,value) in pd.read_csv(path, 
                   nrows=1).iloc[0].to_dict().items() if 'date' in value]
    # Read the rest of the lines with the types from above
    return pd.read_csv(path, dtype=dtypes, parse_dates=parse_dates, skiprows=[1])