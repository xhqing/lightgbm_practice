# lightgbm_practice

Python code in this repo are Python3.9.12, running someone pyfile using:
```bash
pipenv run python xxx.py
```

## Usage of lgb.Dataset()

```python
data = np.random.rand(500, 10)  # 500 entities, each contains 10 features
label = np.random.randint(2, size=500)  # binary target
```

### Specific feature names and categorical features:

```python
train_data = lgb.Dataset(data, label=label, feature_name=['c1', 'c2', 'c3'], categorical_feature=['c3'])
```
LightGBM can use categorical features as input directly. It doesn’t need to convert to one-hot encoding, and is much faster than one-hot encoding (about 8x speed-up).

Note: You should convert your categorical features to int type before you construct Dataset.

### Weights can be set when needed:
```python
w = np.random.rand(500, )
train_data = lgb.Dataset(data, label=label, weight=w)
```
or
```
train_data = lgb.Dataset(data, label=label)
w = np.random.rand(500, )
train_data.set_weight(w)
```
