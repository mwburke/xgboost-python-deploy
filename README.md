# XGBoost Python Deployment

This package allows users to take their XGBoost models they developed in python, and package them in a way that they can deploy the model in production using only pure python.

## Installation

You can install this from pip using `pip install xgboost-deploy-python`. The current version is 0.0.2.

It was tested on python 3.6 and and XGBoost 0.8.1.


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
    - When building a classification model, you **must** use the default base_score value of 0.5 (which ends up not adding an intercept bias to the results). If you use any other value, the production model will produce predictions that **do not match** the predictions from the original model.
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
Mean difference between predictions: -1.270028868389972e-08
Std dev of difference between predictions: 9.327025899795536e-09

Actual Estimator Evaluation Metrics
AUROC Score 0.9780560216858138
Accuracy Score 0.9521276595744681
F1 Score 0.9649805447470817

Prod Estimator Evaluation Metrics:
AUROC Score 0.9780560216858138
Accuracy Score 0.9521276595744681
F1 Score 0.9649805447470817


Time Benchmarks for 1 records with 30 features using 10 trees
Average 3.938e-05 seconds with standard deviation 2.114e-06 per 1 predictions


Benchmark classification modeling
=================================

Actual vs Prod Estimator Comparison
188 out of 188 predictions match
Mean difference between predictions: -1.7196643532927356e-08
Std dev of difference between predictions: 2.7826259523143417e-08

Actual Estimator Evaluation Metrics
AUROC Score 0.9777333161223698
Accuracy Score 0.9468085106382979
F1 Score 0.9609375

Prod Estimator Evaluation Metrics:
AUROC Score 0.9777333161223698
Accuracy Score 0.9468085106382979
F1 Score 0.9609375


Time Benchmarks for 1 records with 30 features using 10 trees
Average 3.812e-05 seconds with standard deviation 1.381e-06 per 1 predictions
```

## License

Licensed under an MIT license.
