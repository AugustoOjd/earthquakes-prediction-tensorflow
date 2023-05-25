import pandas as pd
import tensorflow as tf

train_df = pd.read_csv('model/earthquakes_train_model.csv', index_col=0)
test_df = pd.read_csv('model/earthquakes_test_model.csv', index_col=0)


scale_factor = 1000.0

# Scale the training set's label.
train_df["MAG"] /= scale_factor 

# Scale the test set's label
test_df["MAG"] /= scale_factor

# DEFINE MODEL -----------------------------------------------------------------------------------------

#@title Define the functions that build and train a model
def build_model(my_learning_rate):
  """Create and compile a simple linear regression model."""
  # Most simple tf.keras models are sequential.
  model = tf.keras.models.Sequential()

  # Add one linear layer to the model to yield a simple linear regressor.
  model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

  # Compile the model topography into code that TensorFlow can efficiently
  # execute. Configure training to minimize the model's mean squared error. 
  model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

  return model 