import numpy as np
import pandas as pd
import pytest

from utilities import get_cumulative, get_indicator

@pytest.fixture
def simple_df():
    return pd.DataFrame([['0001','tum1','1'], ['0001','tum2', '2'], ['0001','tum3', '2'],['0002','tum4','1'],['0002','tum5','3']], columns=['PATIENTID','TUMOURID','DATE'])

@pytest.fixture
def dup_df():
    return pd.DataFrame([['0001','tum1','1'], ['0001','tum2', '2'], ['0001','tum2', '2']], columns=['PATIENTID','TUMOURID','DATE'])

@pytest.fixture
def nan_df():
    return pd.DataFrame([['0001','tum1','1','11'], ['0001','tum2','2','11'], ['0001','tum3', '2',np.nan],['0002','tum4','1',np.nan],['0002','tum5','3','13']], columns=['PATIENTID','TUMOURID','DATE','REGDATE'])

@pytest.fixture
def bool_df():
    return pd.DataFrame([['0001','tum1','1',False], ['0001','tum2','2',True], ['0001','tum3', '2',False],['0001','tum4','3',False],['0002','tum5','3',True]], columns=['PATIENTID','TUMOURID','DATE','BOOL'])

@pytest.fixture
def nan_bool_df():
    # return pd.DataFrame([['0001','tum1','1',False], ['0001','tum2','2',True], ['0001','tum3', np.nan,False],['0001','tum4','3',False],['0002','tum5','3',True]], columns=['PATIENTID','TUMOURID','DATE','BOOL'])
    return pd.DataFrame([['0001','tum1','1',False], ['0001','tum1','1',True],['0001','tum2','2',True], ['0001','tum3', np.nan,False],['0001','tum4','3',False],['0002','tum5','3',True]], columns=['PATIENTID','TUMOURID','DATE','BOOL'])

@pytest.fixture
def dup_bool_df():
    return pd.DataFrame([['0001','tum1','1',False], ['0001','tum1','1',False],['0001','tum2','1',False], ['0001','tum2', '1',False]], columns=['PATIENTID','TUMOURID','DATE','BOOL'])


def test_cumulative_handles_same_date(simple_df):

    simple_df = get_cumulative(simple_df, 'PATIENTID', 'DATE', 'OUTPUT')

    expected = pd.DataFrame([[1],[3],[3],[1],[2]], columns=['OUTPUT'])
    print(simple_df)
    assert(simple_df['OUTPUT'].equals(expected['OUTPUT']))

def test_cumulative_handles_duplicates(dup_df):

    dup_df = get_cumulative(dup_df, 'PATIENTID', 'DATE', 'OUTPUT')

    expected = pd.DataFrame([[1],[2],[2]], columns=['OUTPUT'])
    print(dup_df)
    assert(dup_df['OUTPUT'].equals(expected['OUTPUT']))

def test_cumulative_handles_nans(nan_df):

    nan_df = get_cumulative(nan_df, 'PATIENTID', 'REGDATE', 'OUTPUT')
    expected = pd.DataFrame([[2],[2],[np.nan],[np.nan],[1]], columns=['OUTPUT'])
    print(nan_df)
    assert(nan_df['OUTPUT'].equals(expected['OUTPUT']))

def test_indicator_handles_same_date(bool_df):
    bool_df = get_indicator(bool_df, 'BOOL','PATIENTID','DATE','OUTPUT')
    expected = pd.DataFrame([[False],[True],[True],[True],[True]], columns=['OUTPUT'])
    print(bool_df)
    assert(bool_df['OUTPUT'].equals(expected['OUTPUT']))

def test_indicator_handles_nans(nan_bool_df):
    nan_bool_df = get_indicator(nan_bool_df, 'BOOL','PATIENTID','DATE','OUTPUT')
    expected = pd.DataFrame([[True],[True],[True],[np.nan],[True],[True]], columns=['OUTPUT'])
    print(nan_bool_df)
    assert(nan_bool_df['OUTPUT'].equals(expected['OUTPUT']))

def test_indicator_handles_duplicates(dup_bool_df):
    dup_bool_df = get_indicator(dup_bool_df, 'BOOL','PATIENTID','DATE','OUTPUT','TUMOURID')
    expected = pd.DataFrame([[False],[False],[False],[False]], columns=['OUTPUT'])
    print(dup_bool_df)
    assert(dup_bool_df['OUTPUT'].equals(expected['OUTPUT']))