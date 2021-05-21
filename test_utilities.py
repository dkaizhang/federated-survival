import numpy as np
import pandas as pd
import pytest

from utilities import get_cumulative

@pytest.fixture
def simple_df():
    return pd.DataFrame([['0001','tum1','1'], ['0001','tum2', '2'], ['0001','tum3', '2'],['0002','tum4','1'],['0002','tum5','3']], columns=['PATIENTID','TUMOURID','DATE'])

@pytest.fixture
def nan_df():
    return pd.DataFrame([['0001','tum1','1','11'], ['0001','tum2','2','11'], ['0001','tum3', '2',np.nan],['0002','tum4','1',np.nan],['0002','tum5','3','13']], columns=['PATIENTID','TUMOURID','DATE','REGDATE'])

def test_cumulative_handles_same_date(simple_df):

    simple_df = get_cumulative(simple_df, 'PATIENTID', 'DATE', 'OUTPUT','TUMOURID')

    expected = pd.DataFrame([[1],[3],[3],[1],[2]], columns=['OUTPUT'])
    print(simple_df)
    assert(simple_df['OUTPUT'].equals(expected['OUTPUT']))

def test_cumulative_handles_nans(nan_df):

    nan_df = get_cumulative(nan_df, 'PATIENTID', 'REGDATE', 'OUTPUT','TUMOURID')
    expected = pd.DataFrame([[2],[2],[np.nan],[np.nan],[1]], columns=['OUTPUT'])
    print(nan_df)
    assert(nan_df['OUTPUT'].equals(expected['OUTPUT']))
