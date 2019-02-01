from xgb_deploy.fmap import generate_fmap_from_pandas
from xgb_deploy.model import ProdEstimator
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from pprint import pprint
import xgboost as xgb
import pandas as pd
import numpy as np
import json


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


cancer = load_breast_cancer()

df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

# Replace space from column names to avoid JSON model dump errors
df.columns = [c.replace(' ', '_') for c in df.columns]

# Convert continuous fields to integer fields
for col in ['worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area']:
    df[col] = df[col].astype(int)

# Convert continuous fields to binary fields
for col in ['worst_smoothness', 'worst_compactness', 'worst_concavity']:
    df[col] = (df[col] < np.median(df[col])).astype(int)

# Generate fmap file
generate_fmap_from_pandas(df, 'fmap_pandas.txt')

feature_cols = [c for c in df.columns if c != 'target']

x_train, x_test, y_train, y_test = train_test_split(df[feature_cols], df['target'], test_size=0.33)

dtrain = xgb.DMatrix(data=x_train, label=y_train)
dtest = xgb.DMatrix(data=x_test)

params = {
    'base_score': np.mean(y_train),
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'silent': 1
}

pred_type = 'classification'

boost_rounds = 3

model = xgb.train(params=params, dtrain=dtrain, num_boost_round=boost_rounds)

model.dump_model(fout='xgb_model.json', fmap='fmap_pandas.txt', dump_format='json')

with open('xgb_model.json', 'r') as f:
    model_data = json.load(f)

estimator = ProdEstimator(model_data, pred_type, params['base_score'])

predictions = model.predict(dtest)
data = x_test.to_dict(orient='records')
prod_predictions = estimator.predict(data)

tolerance = 0.001

i = 0
for p, pp in zip(predictions, prod_predictions):
    if np.abs(p - pp) > tolerance:
        break
    i += 1

print('Debugging Predicted Probability Calculations')
print('============================================')
print()
print('Example for a single input:')
print()
print('Actual prediction:', p)
print('Production prediction:', pp)
print()
print('Actual Model Leaf Nodes')
print(model.predict(data=xgb.DMatrix(data=x_test.iloc[i:i + 1, :]), pred_leaf=True)[0])
print()
print('Actual Model Margin Prediction')
print(model.predict(data=xgb.DMatrix(data=x_test.iloc[i:i + 1, :]), output_margin=True)[0])
print()
print('Production Leaf Nodes')
print(np.array(estimator.get_leaf_values(data[i], False)))
print()
print('Production Leaf Values')
leaf_values = estimator.get_leaf_values(data[i])
print(np.array(leaf_values))
print()
print('Sum of leaf values:', np.sum(leaf_values))
print('Base score:', params['base_score'])
print()
print('Testing different calculation methods:')
print('Sigmoid of (leaf value sum):', sigmoid(np.sum(leaf_values)))
print('Sigmoid of (leaf value sum + base score):', sigmoid(np.sum(leaf_values) + params['base_score']))
print('Sigmoid of (leaf value sum - base score):', sigmoid(np.sum(leaf_values) - params['base_score']))
print('Sigmoid of (leaf value sum) - base score:', sigmoid(np.sum(leaf_values)) - params['base_score'])
print('Sigmoid of (leaf value sum) + base score:', sigmoid(np.sum(leaf_values)) + params['base_score'])
print()
# print('Data')
# pprint(data[i])
# print()
# print('Decision Trees')
# pprint(model_data)
