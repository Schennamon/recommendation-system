import pandas as pd
from sklearn.model_selection import train_test_split
from caserec.recommenders.rating_prediction.userknn import UserKNN
from sklearn.metrics import mean_squared_error
from math import sqrt

df = pd.read_csv('../datasets/AMAZON_FASHION.csv', nrows=100000)
df.columns = ['item', 'user', 'rating', 'timestamp']

user_ids = df['user'].unique().tolist()
user_id_to_num = {user_id: i for i, user_id in enumerate(user_ids)}

item_ids = df['item'].unique().tolist()
item_id_to_num = {item_id: i for i, item_id in enumerate(item_ids)}

df['user'] = df['user'].map(user_id_to_num)
df['item'] = df['item'].map(item_id_to_num)

train_data, test_data = train_test_split(df[['user', 'item', 'rating']], test_size=0.2)
train_data.to_csv('train.csv', index=False, header=False)
test_data.to_csv('test.csv', index=False, header=False)

similarity_metric = 'cosine'
k_neighbors = 20

model = UserKNN(train_file='train.csv',
                test_file='test.csv',
                output_file='output_predictions.csv',
                similarity_metric=similarity_metric,
                k_neighbors=k_neighbors,
                sep=',')

try:
    print("Starting model training and prediction...")
    model.compute()
    print("Model training and prediction completed. Check the output file.")
except Exception as e:
    print("Error occurred:", e)

predictions = pd.read_csv('output_predictions.csv', header=None, names=['user', 'item', 'rating_predicted'])
test_data = pd.read_csv('test.csv', header=None, names=['user', 'item', 'rating_actual'])

merged = pd.merge(test_data, predictions, on=['user', 'item'], how='inner')

if not merged.empty:
    rmse = sqrt(mean_squared_error(merged['rating_actual'], merged['rating_predicted']))
    print(f'RMSE: {rmse}')
else:
    print('Merged dataframe is empty. No common data to compare.')
