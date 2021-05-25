import pandas as pd

# calculates a cumulative count of the target column based on a date column
# first project only targets and date column
# collapse based on key to remove duplicates in the working table
# DOES not remove duplicates in original input table
def get_cumulative(table, groupby, date, name, key=None):
    
    df = table.copy()
    start = df.shape[0]

    if type(groupby) is not list:
        groupby = [groupby]
    
    filterby = groupby + [date]
    # collapse the dataframe based on keys
    if key is not None:
        if type(key) is not list:
            key = [key]
        filterby = filterby + key
    temp = df.drop_duplicates(key)[filterby]

    # counts number of records having the same values in groupby + [date]
    temp = temp.groupby(filterby, dropna=False).size().reset_index(name='no_of_records') # reset_index to turn the Series into a DataFrame with columns filled in

    # drop records without date to exclude from cumsum
    # cumulatively sum over number of records
    # dropna=False AND left join to keep all nans in the left table
    temp[name] = temp.dropna(subset=[date]).sort_values(by=date, ascending=True).groupby(groupby)['no_of_records'].cumsum() 

    df = pd.merge(df, temp, on=filterby, how='left')
    print(df)
    df = df.drop(columns=['no_of_records'])

    end = df.shape[0]
    print("lost: ", start - end)
    return df


def get_indicator(table, target, groupby, date, name, key=None):

    df = table.copy()
    start = df.shape[0]

    if type(groupby) is not list:
        groupby = [groupby]
    
    filterby = groupby + [date, target]
    # collapse the dataframe based on keys
    if key is not None:
        if type(key) is not list:
            key = [key]
        filterby = filterby + key
    temp = df.drop_duplicates(key)[filterby]

    temp['temp1'] = temp[target].astype(float)
    temp['temp2'] = temp.sort_values(by='temp1',ascending=False).sort_values(by=date, ascending=True).groupby(groupby)['temp1'].cumsum()
    # print(temp)
    # temp = temp.drop(columns=target)
    temp[name] = temp.dropna(subset=[date,target])['temp2'] > 0

    df = pd.merge(df, temp, on=filterby, how='left')

    df = df.drop(columns=['temp1','temp2'])

    end = df.shape[0]
    print("lost: ", start - end)
    return df

def merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):
    left = left.merge(right,how,on,left_on,right_on,left_index,right_index,sort,suffixes,copy,indicator,validate)
    drop = ['_r' in c for c in left.columns]
    return left.drop(columns=left.columns[drop])

def add_aggregate(table,what,target,groupby,name,key=None):

    df = table.copy()
    if groupby is not list:
        groupby = [groupby]

    filterby = groupby + [target]

    if key is not None:
        if type(key) is not list:
            key = [key]
        filterby = filterby + key

    df = df.drop_duplicates(key)[filterby]

    nonnulls = ~df[target].isnull()
    df[name] = df[nonnulls].groupby(groupby)[target].transform(what)
    df[name] = df[[name] + groupby].groupby(groupby)[name].transform('max')

    return df