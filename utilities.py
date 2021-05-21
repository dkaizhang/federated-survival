import pandas as pd

# calculates a cumulative count of the target column based on a date column
# first project only targets and date column
# collapse to remove duplicates

def get_cumulative(table, groupby, date, name, key=None):
    
    df = table.copy()
    start = df.shape[0]

    if type(groupby) is not list:
        groupby = [groupby]
    
    # collapse the dataframe based on keys
    if key is not None:
        if type(key) is not list:
            key = [key]
    temp = df.drop_duplicates(key)[groupby + [date]]

    # counts number of records having the same values in groupby + [date]
    # drop records without date
    # sort by date 
    # group by groupby
    # then cumulatively sum over number of records
    # dropna=False AND left join to keep all nans in the left table
    temp = temp.groupby(groupby+[date], dropna=False).size().reset_index(name='no_of_records')
    temp[name] = temp.dropna(subset=[date]).sort_values(by=date, ascending=True).groupby(groupby)['no_of_records'].cumsum()

    df = pd.merge(df, temp, on=groupby+[date], how='left')

    df = df.drop(columns=['no_of_records'])

    end = df.shape[0]
    print("lost: ", start - end)
    return df


def get_indicator(table, target, groupby, date, name, key=None):

    df = table.copy()
    start = df.shape[0]

    if type(groupby) is not list:
        groupby = [groupby]
    
    # collapse the dataframe based on keys
    temp = df
    if key is not None:
        if type(key) is not list:
            key = [key]
        temp = temp.drop_duplicates(key)[groupby + key +[date,target]]
    else: 
        temp = temp[groupby + [date,target]]
    temp['temp1'] = temp[target].astype(float)
    temp['cum'] = temp.sort_values(by='temp1',ascending=False).sort_values(by=date, ascending=True).groupby(groupby)['temp1'].cumsum()
    print(temp)
    # temp = temp.drop(columns=target)
    temp[name] = temp.dropna(subset=[date,target])['cum'] > 0

    if key is not None:
        df = pd.merge(df, temp, on=groupby+key+[date,target], how='left')
    else:
        df = pd.merge(df, temp, on=groupby+[date,target], how='left')

    df = df.drop(columns=['temp1','cum'])

    end = df.shape[0]
    print("lost: ", start - end)
    return df
