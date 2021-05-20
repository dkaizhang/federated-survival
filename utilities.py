import pandas as pd

# calculates a cumulative count of the target column based on a date column
def get_cumulative(table, target, date, name):
    df = table.copy()
    df['temp1'] = df.sort_values(by=date, ascending=True).groupby([target]).cumcount() + 1
    temp = df.groupby([target, date]).size().reset_index(name='temp2')
    df = pd.merge(df, temp, on=[target, date])
    df[name] = df[['temp1', 'temp2']].max(axis=1)
    df = df.drop(columns=['temp1', 'temp2'])
    return df

