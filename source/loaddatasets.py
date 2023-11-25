import pandas as pd
from sklearn.utils import shuffle

def transform_label(value):
    if value == 6:
        return 0
    elif value == 5 :
        return 1
    else:
        return 2

def iterabel_dataset_generation(path):
    df = pd.read_csv(path)
    df['new_label'] = df['new_label'].apply(transform_label)
    sensor_data = df.filter(regex='^Sensor')
    labels = df['new_label']


    label_dummies = pd.get_dummies(labels).astype(int)
    label_dummies.columns = [f'Label{i}-fail' for i in range(label_dummies.shape[1])]

    index = df['new_label'].values

    data = df.iloc[:, :-1]
    num_rows, num_cols = data.shape
    rows_per_chunk = num_rows // 10
    cols_per_chunk = num_cols // 10

    for i in range(9):
        start_row = i * rows_per_chunk
        end_row = start_row + rows_per_chunk
        end_col = (i + 1) * cols_per_chunk


        data.iloc[start_row:end_row, end_col:] = 0


    data.iloc[9 * rows_per_chunk:, :] = data.iloc[9 * rows_per_chunk:, :]

    label = df.iloc[:, -1].values.astype(int)

    data, label, index = shuffle(data, label, index, random_state=1020)

    return data.values, label, index
