import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
#Eager execution
tfe.enable_eager_execution()

a = tf.constant([1,2,3])
b = tf.constant([2,3,4])
print(a-b)
