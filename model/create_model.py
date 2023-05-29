import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

train_df = pd.read_csv('model/earthquakes_train_model.csv', index_col=0)
test_df = pd.read_csv('model/earthquakes_test_model.csv', index_col=0)

train_df = train_df.reindex(np.random.permutation(train_df.index)) # shuffle the training set

print(train_df.head())

# scale_factor = 1000.0

# # Scale the training set's label.
# train_df["MAG"] /= scale_factor 

# # Scale the test set's label
# test_df["MAG"] /= scale_factor

# # DEFINE MODEL -----------------------------------------------------------------------------------------

