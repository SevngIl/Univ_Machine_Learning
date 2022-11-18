# Machine_Learning_team7
Machine learning subject PHW1 assignment results
---

# Main task

We will show 4 clustering algorithms using python. 

- Decision Tree Classifier using entropy
- Decision Tree classifier using gini
- Logistifc Regression
- SVC

# Data
![image](https://user-images.githubusercontent.com/51481256/195965346-7388f63d-5726-4ddb-928d-9de18ba1f47e.png)

Dataset = Breast Cancer Wisconsin (Diagnostic) Data Set

# Prepare scalers and classification models

Our model's scaler would be
  * StandardScaler
  * MinMaxScaler
  * RobustScaler

and each model would use each scaler every try like this.

![image](https://user-images.githubusercontent.com/51481256/195965927-40e0c256-9b91-4431-a2fb-ceec0c905860.png)

![image](https://user-images.githubusercontent.com/51481256/195966341-3c8a4f71-d8b3-4551-b8b4-d784d8b05ac2.png)

Our Classification models are Decision tree(by using entropy, by using Gini index), Logistic regression, SVM and we set a parameter and hyperparameters like this

![image](https://user-images.githubusercontent.com/51481256/195965827-0392f81f-7177-4464-abb3-7e9634e17b6f.png)

# Parameters of models in python

---

## DecisionTreeClassifier using entropy

```python
DecisionTreeClassifier() : {
            'criterion' : ['entropy'],
            'max_depth' : [1,3,5,7,9],
            'min_samples_leaf' : [2,5,7,9]
        }
```

## DecisionTreeClassifier using gini

```python
DecisionTreeClassifier() : {
            'criterion' : ['gini'],
            'max_depth' : [1,3,5,7,9],
            'min_samples_leaf' : [2,5,7,9]
        }
```


## LogisticRegression

```python
LogisticRegression() : {
            'solver': ['newton-cg','lbfgs', 'liblinear'],
            'penalty': ['l2'],
            'C': [0.01, 0.1, 1.0, 10, 100],
        }
```

## SVC

```python
SVC() : {
            'kernel': ['poly', 'rbf', 'sigmoid'],
            'C': [0.01, 0.1, 1.0, 10, 100],
            'gamma': ['scale']
        }
```

## K fold

```python
Ks = [5, 7, 10]
```


# Show results

By using this way, we run different combinations of the data scailing methods and encoding methods, model parameters, hyperparameters, k for k-fold cross validaiton.

![image](https://user-images.githubusercontent.com/51481256/195966538-ec122b77-add1-445a-bd9f-f0725cbd44b7.png)

* GridSearchCV: Determines the cross-validation splitting strategy.

![image](https://user-images.githubusercontent.com/51481256/195968732-fcc5f025-9ae0-4803-8ca3-55ff430d805a.png)

![image](https://user-images.githubusercontent.com/51481256/195968744-f8536b23-e628-4d73-af39-9965263fb827.png)

K value would be a 5, 7, 10 and we can find out Best Score and Parameter Value when each setting.
