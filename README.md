
# AdaBoost Implementation

Implementation of AdaBoost with "optimal" decision stumps on the training data. After each round, the following gets computed:
    
1. Current training error for the weighted linear combination predictor at this round
2. Current testing error for the weighted linear combination predictor at this round
3. Current test AUC for the weighted linear combination predictor at this round
4. The local "round" error for the decision stump returned

### Supported python versions

Python 3

## Documentation

### Load the spambase dataset and split into train and test


```python
from Datasets import spambase
filename = "data/Spambase dataset/spambase.data"
train_X, train_y, test_X, test_y = spambase(filename)
```

### Setup model (following parameters are default)


```python
from AdaBoost import AdaBoost
model = AdaBoost(iterations = 100)
```

### Train model


```python
model.fit(train_X, train_y, test_X, test_y)
```

### Plot of train and test error versus number of iterations


```python
model.plot_train_test_error()
```

### Plot of final ROC curve


```python
model.plot_ROC_curve()
```

### Plot of local round error which reduces after each iteration


```python
model.plot_round_error()
```

## Results

Train error, Test error and Test AUC after every 25 iterations:

 - Round 0   : __Train_err__: 0.20760869565217388 __Test_err__: 0.21064060803474483 __AUC__: 0.748974795114
 - Round 25  : __Train_err__: 0.06766304347826091 __Test_err__: 0.07600434310532034 __AUC__: 0.978207515077
 - Round 50  : __Train_err__: 0.060054347826086985 __Test_err__: 0.07057546145494031 __AUC__: 0.982347610948
 - Round 75  : __Train_err__: 0.056793478260869557 __Test_err__: 0.06297502714440828 __AUC__: 0.984188340807
 - Round 100 : __Train_err__: 0.05461956521739131 __Test_err__: 0.061889250814332275 __AUC__: 0.985246946034

Graph of Train/Test Error, ROC Curve, Round Error vs number of iterations:

![title](img/output_12_0.png)

![title](img/output_14_0.png)

![title](img/output_16_0.png)
