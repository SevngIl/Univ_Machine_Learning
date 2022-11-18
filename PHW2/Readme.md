# 2022 ML PHW 2

---

This code is the result of 2022 Machine Learning class PHW.

# Main task

We will show 4 clustering algorithms using python. 

- K-means
- EM(GMM)
- CLARANS
- DBSCAN

# Dataset

[California Housing Prices](https://www.kaggle.com/camnugent/california-housing-prices)

We used california Housing prices dataset.  

You can download this dataset from above link. 

# Function 1: Data reading

---

```python
def data_reading()
```

- Original dataset is csv. We will use `pandas read_csv` and transform it into Dataframe. Next, by using `pandas head()` , we will check basic features of dataset.

# Function 2: Preprocessing

---

## Preprocessing

```python
def preprocessing(dataset, encoder, scaling, encode_feature_list):
```

- ***dataset*** : we changed dataset into pandas datafram form.
- ***encoder***: To use encoding function, we need list type of encoders.
- ***scaling***: Dataset should be scaled before putting it into models. So, like encoder, we get lists of scalers.
- ***encode_feature_list***: We get feature list that need encoding.

## Encoding

```python
def object_encoder(dataframe, encoder, target_feature)
```

- ***dataframe*** : Dataset that has form of pandas dataframe
- ***encoder***: Some features are categorical data, so by this parameter we can change categorical data into numerical data. we get list[ ] types so that we can experience various types of encoders.
- ***Target_feature***: We get feature list that need encoding.

## Scaling

```python
def data_scaling(dataframe, scaling):
```

- ***dataframe*** : Dataset that has form of pandas dataframe
- ***Scaling***: get scaler that we will use.

# Function 3: K-means

---

## Kmeans_cluster

```python
def kmeans_cluster(X_features, scaler)
```

- ***X_freatures*** : features that will be used for clustering
- ***scaler*** : get scaler that we will use.

This function visualizes the area of each silhouette coefficient of the number of clusters. 

## Kmeans_plot

```python
def K_means_plot(df)
```

- ***df*** : Dataset that will be used for k-means clustering.

This function visualizes the result of K-means result by using `plt( )`.

# Function 4: Em cluster

---

## Em cluster

```python
def em_cluster(df, scaler)
```

- ***df*** : Dataset that will be used for EM clustering.
- ***scaler*** : get scaler that we will use.

## Gmm silhouette

```python
def gmm_silhouette(cluster_lists, X_features, scaler)
```

- ***cluster_lists*** : lists of cluster that will be shown by silhoutte score.
- ***X_freatures*** : features that will be used for clustering.
- ***scaler*** : get scaler that we will use.

## Plot Gmm

```python
def plot_gmm(gmm, X, label=True, ax=None)
```

- ***gmm*** : get the `gmm` model.
- ***X*** : features that will be used for clustering.

Show result of Gmm silhouette.

# Function 5: Clarans cluster

---

```python
def clarans_cluster(df, scaler)
```

- ***df*** : Dataset that will be used for EM clustering.
- ***scaler*** : get scaler that we will use.

This function shows the result of clarans clustering. But, this algorithm takes very long time when using all dataset, so we shinked dataset size and applied to this algorithm. 

# Function 6: DBSCAN cluster

---

```python
def DBSCAN_cluster(df,scaler)
```

- ***df*** : Dataset that will be used for EM clustering.
- ***scaler*** : get scaler that we will use.

This function shows the result of DBSCAN clustering. 

# Function 7: AutoML

---

```python
def auto_ml(input_dataset, model_lists, encoder_lists, scaling_lists, select_feature_lists, k_lists=None)
```

- input_dataset : Dataset that will be use for clustering
- model_lists: lists of cluster models
- encoder_lists:  lists of encoders
- scaling_lists: lists of scalers
- select_feature_lists: get the feature lists

This function helps automatically clusters various algorithms.