# 30_days_30_models
Assortment of TensorFlow and Numpy models. One a day for 30 days. Loosely follows Scikit-Learn API. For personal learning, not production. Will refactor as I go. Starting with more traditional machine learning models.

Factorization Machines:
----------------------
TensorFlow implementation of the original [paper](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) by Steffen Rendel.
```python
from fm.model import FactorizationMachines
from utils import generate_rendle_style_dataset

## Obtain data
x_data, y_data = generate_rendle_style_dataset()

## Fit and predict
fm = FactorizationMachines(l_factors=10)
fm.fit(x_data, y_data)
fm.predict(x_data)
```

Logistic Regression:
--------------------
A multi-class, Numpy implementation with l2-norm regularization:
```python
from lm.np_model import LogisticRegression
from utils import generate_classification_style_dataset

## Obtain data
X, Y = generate_classification_style_dataset()

## Fit and predict
lr = LogisticRegression()
lr.fit(X, Y)
lr.predict(X)
```
To-Do:
------
In no particular order:
- [ ] Word2Vec
- [ ] Word Mover's Distance
- [ ] Sequence2Sequence models
- [ ] Singular Value Decomposition & Latent Semantic Indexing
- [ ] Bayesian Linear Regression
- [ ] K-means
- [ ] T-SNE
- [ ] Conditional Random Fields
- [ ] StarSpace
- [ ] Metric Learning models
- [ ] Decision Tree
- [ ] Random Forests
