import numpy as np
import pandas as pd
import pytest

from utilities import get_cumulative, get_indicator, add_aggregate

# get cumulative
@pytest.fixture
def simple_df():
    return pd.DataFrame([['0001','tum1','1'], ['0001','tum2', '2'], ['0001','tum3', '2'],['0002','tum4','1'],['0002','tum5','3']], columns=['PATIENTID','TUMOURID','DATE'])

@pytest.fixture
def dup_df():
    return pd.DataFrame([['0001','tum1','1'], ['0001','tum2', '2'], ['0001','tum2', '2']], columns=['PATIENTID','TUMOURID','DATE'])

@pytest.fixture
def nan_df():
    return pd.DataFrame([['0001','tum1','1','11'], ['0001','tum2','2','11'], ['0001','tum3', '2',np.nan],['0002','tum4','1',np.nan],['0002','tum5','3','13']], columns=['PATIENTID','TUMOURID','DATE','REGDATE'])

# get indicator
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

# add aggregate
@pytest.fixture
def varying_heights_df():
    return pd.DataFrame([['0001','tum1',2.0], ['0001','tum2',3.0],['0002','tum3',np.nan],['0003','tum3',1.0],['0003','tum3',np.nan]], columns=['PATIENTID','TUMOURID','HEIGHT'])
    

def test_cumulative_handles_same_date(simple_df):

    df = get_cumulative(simple_df, 'PATIENTID', 'DATE', 'OUTPUT')

    expected = pd.DataFrame([[1],[3],[3],[1],[2]], columns=['OUTPUT'])
    print(df)
    assert(df['OUTPUT'].equals(expected['OUTPUT']))

def test_cumulative_handles_duplicates(dup_df):

    df = get_cumulative(dup_df, 'PATIENTID', 'DATE', 'OUTPUT')

    expected = pd.DataFrame([[1],[2],[2]], columns=['OUTPUT'])
    print(df)
    assert(df['OUTPUT'].equals(expected['OUTPUT']))

def test_cumulative_handles_nans(nan_df):

    df = get_cumulative(nan_df, 'PATIENTID', 'REGDATE', 'OUTPUT')
    expected = pd.DataFrame([[2],[2],[np.nan],[np.nan],[1]], columns=['OUTPUT'])
    print(df)
    assert(df['OUTPUT'].equals(expected['OUTPUT']))

def test_indicator_handles_same_date(bool_df):
    df = get_indicator(bool_df, 'BOOL','PATIENTID','DATE','OUTPUT')
    expected = pd.DataFrame([[False],[True],[True],[True],[True]], columns=['OUTPUT'])
    print(df)
    assert(df['OUTPUT'].equals(expected['OUTPUT']))

def test_indicator_handles_nans(nan_bool_df):
    df = get_indicator(nan_bool_df, 'BOOL','PATIENTID','DATE','OUTPUT')
    expected = pd.DataFrame([[True],[True],[True],[np.nan],[True],[True]], columns=['OUTPUT'])
    print(df)
    assert(df['OUTPUT'].equals(expected['OUTPUT']))

def test_indicator_handles_duplicates(dup_bool_df):
    df = get_indicator(dup_bool_df, 'BOOL','PATIENTID','DATE','OUTPUT','TUMOURID')
    expected = pd.DataFrame([[False],[False],[False],[False]], columns=['OUTPUT'])
    print(df)
    assert(df['OUTPUT'].equals(expected['OUTPUT']))

def test_aggregate_returns_median(varying_heights_df):
    df = add_aggregate(varying_heights_df,'median','HEIGHT','PATIENTID','OUTPUT')
    expected = pd.DataFrame([[2.5],[2.5],[np.nan],[1.0],[1.0]], columns=['OUTPUT'])
    print(df)
    assert(df['OUTPUT'].equals(expected['OUTPUT']))

