# XGBoost Python Deployment

This package allows users to take their XGBoost models they developed in python, and package them in a way that they can deploy the model in production using only pure python.

**IMPORTANT NOTE:** currently, this package seems to be working correctly for regression problems, but is having consistent mismatches between the original and production versions of the models using the `binary:logistic` objective. This is being raised as an issue directly with XGBoost to figure out how the tree leaf values are being generated in this case. For a more detailed look into this issue, please check out the `debug.py` file. Interestingly enough, the AUROC from original and production models are equal on the same data, which indicates that the transformations performed on both maintain overall prediction ordering but causes errors in terms of true magnitude.

## Installation

You can install this from pip using `pip install pip install xgboost-deploy-python`. The current version is 0.0.1.

It was tested on python 3.6 and and XGBoost 0.81.

## License

© Contributors, 2019. Licensed under an [Apache-2](https://github.com/mwburke/xgboost-deploy-python/blob/master/LICENSE) license.


## Deployment Process

The typical way of moving models to production is by using the `pickle` library to export the model file, and then loading it into the container or server that will make the predictions using `pickle` again. Both the training and deployment environments require the same package and versions of those packages instsalled for this process to work correctly.

This package follows a similar process for deployment, except that it relies only on a single JSON file, and the pure python code that makes predictions, removing the dependencies of `XGBoost` and `pickle` in deployment.

Below we will step through how to use this package to take an existing model and generate the file(s) needed for deploying your model. For a more detailed walk-through, please see `example.py`, which you can run locally to create your own files and view some potential steps for validating the consistency of your trained model vs the new production version.

### `fmap` Creation

In this step we use the `fmap` submodule to create a feature map file that the module uses to accurately generate the JSON files.

A `fmap` file format contains one line for each feature according to the following format: `<line_number> <feature_name> <feature_type>`

Some notes about the full file format:

* `line_number` values start at zero and increase incrementally.
* `feature_name` values **cannot** contain spaces in them or the file will truncate the feature names after the first space in the JSON model file
* `feature_type` options are as follows:
    - `q` for quantitative (or cintuous) variables
    - `int` for integer valued variables
    - `i` for binary variables, please note that this field does **not** allow null values and always expects either a 0 or a 1. Your predictor function may error and fail.

Here is an `fmap` file generated using the `example.py` file:

```
0 mean_radius q
1 mean_texture q
2 mean_perimeter q
3 mean_area q
4 mean_smoothness q
5 mean_compactness q
6 mean_concavity q
7 mean_concave_points q
8 mean_symmetry q
9 mean_fractal_dimension q
10 radius_error q
11 texture_error q
12 perimeter_error q
13 area_error q
14 smoothness_error q
15 compactness_error q
16 concavity_error q
17 concave_points_error q
18 symmetry_error q
19 fractal_dimension_error q
20 worst_radius int
21 worst_texture int
22 worst_perimeter int
23 worst_area int
24 worst_smoothness i
25 worst_compactness i
26 worst_concavity i
27 worst_concave_points q
28 worst_symmetry q
29 worst_fractal_dimension q
30 target i
```

You can either generate your `fmap` file using lists of feature names and types using the `generate_fmap` function or automatically from a `pandas` DataFrame using the `generate_fmap_from_pandas` which extracts column names and infers feature types.

### JSON Model Creation

The `XGBoost` package already contains a method to generate text representations of trained models in either text or JSON formats.

Once you have the `fmap` file created successfully and your model trained, you can generate the JSON model file directly using the following command:

```python
model.dump_model(fout='xgb_model.json', fmap='fmap_pandas.txt', dump_format='json')
```

This should generate a JSON file in your `fout` location specified that should have a list of JSON objects, each representing a tree in your model. Here's an example of a subset of one such file:

```json
[
  { "nodeid": 0, "depth": 0, "split": "worst_perimeter", "split_condition": 110, "yes": 1, "no": 2, "missing": 1, "children": [
    { "nodeid": 1, "depth": 1, "split": "worst_concave_points", "split_condition": 0.160299987, "yes": 3, "no": 4, "missing": 3, "children": [
      { "nodeid": 3, "depth": 2, "split": "worst_concave_points", "split_condition": 0.135049999, "yes": 7, "no": 8, "missing": 7, "children": [
        { "nodeid": 7, "leaf": 0.150075898 },
        { "nodeid": 8, "leaf": 0.0300908741 }
      ]},
      { "nodeid": 4, "depth": 2, "split": "mean_texture", "split_condition": 18.7449989, "yes": 9, "no": 10, "missing": 9, "children": [
        { "nodeid": 9, "leaf": -0.0510330871 },
        { "nodeid": 10, "leaf": -0.172740772 }
      ]}
    ]},
    { "nodeid": 2, "depth": 1, "split": "worst_texture", "split_condition": 20, "yes": 5, "no": 6, "missing": 5, "children": [
      { "nodeid": 5, "depth": 2, "split": "mean_concave_points", "split_condition": 0.0712649971, "yes": 11, "no": 12, "missing": 11, "children": [
        { "nodeid": 11, "leaf": 0.099997662 },
        { "nodeid": 12, "leaf": -0.142965034 }
      ]},
      { "nodeid": 6, "depth": 2, "split": "mean_concave_points", "split_condition": 0.0284200013, "yes": 13, "no": 14, "missing": 13, "children": [
        { "nodeid": 13, "leaf": -0.0510330871 },
        { "nodeid": 14, "leaf": -0.251898795 }
      ]}
    ]}
  ]},
  {...}
]
```

### Model Predictions

Once the JSON file has been created, you need to perform three more things in order to make predictions:

1. Store the base score value used in training your XGBoost model
2. Note whether your problem is a classification or regression problem.
    - Right now, this package has only been tested for the `reg:linear` and `binary:logistic` objectives which represent regression and classification respectively.
    - The only difference is that classification problems perform a sigmoid transformation on the margin values.
3. Load the JSON model file into a python list of dictionaries representing each model tree.

Once you have done that, you can create your production estimator:

```python
with open('xgb_model.json', 'r') as f:
    model_data = json.load(f)

pred_type = 'regression'
base_score = 0.5  # default value


estimator = ProdEstimator(model_data=model_data,
                          pred_type=pred_type,
                          base_score=base_score)
```

After that, all you have to do is format your input data into python dicts and pass them in individually to the estimator's `predict` function.

If you want more detailed info for validation purposes, there is a `get_leaf_values` function that can return either the leaf values or final nodes selected for each tree in the model for a given input.

## Performance

As mentioned at the top, the initial test results indicate that for regression problems, there is a 100% match in predictions between original and production versions of the models up to a 0.001 error tolerance.

As for speed, for 10 trees, the prediction for input with 30 features is 0.00005 seconds with 0.000008 seconds standard deviation.

Obviously as the number of trees grows, the speed should decrease linearly, but it should be simple to modify this to add in parallelized tree predictions if that becomes an issue.

If you are really looking for optimized deployment tools, I would check out the following compiler for ensemble decision tree models: https://github.com/dmlc/treelite

## Initial Test Results

Here is the printed output for initial testing in the `example.py` file if you want to see it without running it yourself:

```
Benchmark regression modeling
=============================

Actual vs Prod Estimator Comparison
188 out of 188 predictions match
Mean difference between predictions: 4.4960751236712825e-08
Std dev of difference between predictions: 1.4669225492743017e-08

Actual Estimator Evaluation Metrics
AUROC Score 0.9796944858420268
Accuracy Score 0.9202127659574468
F1 Score 0.9407114624505928

Prod Estimator Evaluation Metrics:
AUROC Score 0.9796944858420268
Accuracy Score 0.9202127659574468
F1 Score 0.9407114624505928

Comparison of 5 Predictions
Prediction 0, Actual Model vs Production
0.861517071723938 vs 0.8615170143624671
Prediction 1, Actual Model vs Production
0.21696722507476807 vs 0.21696719166246714
Prediction 2, Actual Model vs Production
0.861517071723938 vs 0.8615170143624671
Prediction 3, Actual Model vs Production
0.6107258200645447 vs 0.6107257820624672
Prediction 4, Actual Model vs Production
0.7350912094116211 vs 0.7350911622924672

Time Benchmarks for 1 records with 30 features using 10 trees
Average 5.4059999999999994e-05 seconds with standard deviation 8.236528394900365e-06 per 1 predictions


Benchmark classification modeling
=================================

Actual vs Prod Estimator Comparison
0 out of 188 predictions match
Mean difference between predictions: 0.18719403647682603
Std dev of difference between predictions: 0.033918844810309955

Actual Estimator Evaluation Metrics
AUROC Score 0.9815573770491804
Accuracy Score 0.9414893617021277
F1 Score 0.9561752988047808

Prod Estimator Evaluation Metrics:
AUROC Score 0.9815573770491804
Accuracy Score 0.9468085106382979
F1 Score 0.9586776859504132

Comparison of 5 Predictions
Prediction 0, Actual Model vs Production
0.85405033826828 vs 0.662387329307486
Prediction 1, Actual Model vs Production
0.2504895031452179 vs 0.10076255829216352
Prediction 2, Actual Model vs Production
0.8517128229141235 vs 0.6582086490048578
Prediction 3, Actual Model vs Production
0.3893463909626007 vs 0.17612317754562867
Prediction 4, Actual Model vs Production
0.8091333508491516 vs 0.5870082924902191

Time Benchmarks for 1 records with 30 features using 10 trees
Average 5.945e-05 seconds with standard deviation 1.533321557925799e-05 per 1 predictions
```

## Initial Classification Debugging Results



```
Debugging Predicted Probability Calculations
============================================

Example for a single input:

Actual prediction: 0.714256
Production prediction: 0.4559565177126631

Actual Model Leaf Nodes
[7 7 7]

Actual Model Margin Prediction
0.9161452

Production Leaf Nodes
[7 7 7]

Production Leaf Values
[0.1548744  0.14447486 0.14081691]

Sum of leaf values: 0.440166175
Base score: 0.6167979002624672

Testing different calculation methods:
Sigmoid of (leaf value sum): 0.6082986262101555
Sigmoid of (leaf value sum + base score): 0.7421099488091216
Sigmoid of (leaf value sum - base score): 0.4559565177126631
Sigmoid of (leaf value sum) - base score: -0.008499274052311656
Sigmoid of (leaf value sum) + base score: 1.2250965264726226
```
