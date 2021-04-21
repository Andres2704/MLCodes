import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

# Importing the MNIST data
mnist = tf.keras.datasets.mnist

# Loading the dataa
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Scaling the data and analysing the shape
x_train, x_test = x_train / 255.0, x_test / 255.0

for i in X_train[0]:
  print(list(i))

# Analysing the shapes
print('Data shape: ', x_train.shape)
print('{} images of size {}x{} px'.format(x_train.shape[0], x_train.shape[1], x_train.shape[1]))

# Analysing the target shape and values
print(y_train.shape)

# Building the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),                                 
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2), # Dropout probability: means that every time we go into the NN layer there is a 20% of drop (Setting to zero) a node in that layer
  tf.keras.layers.Dense(10, activation='softmax') # We have k-classifier, where k = 10, digits from 0 to 9
])
# Configurating the model for training
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)

# Plotting the loss results
import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'valloss')
plt.legend()

# Plotting the loss results
plt.plot(r.history['accuracy'], label = 'accuracy')
plt.plot(r.history['val_accuracy'], label = 'valaccuracy')
plt.legend()

# Evaluate the model
model.evaluate(x_test, y_test)

# Plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()


p_test = model.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))

# Do these results make sense?
# It's easy to confuse 9 <--> 4, 9 <--> 7, 2 <--> 7, etc.

print(model.predict(x_test).argmax(axis = 1))

print(model.predict(x_test)[0])

# Show some misclassified examples
misclassified_idx = np.where(p_test != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i], cmap='gray')
plt.title("True label: %s Predicted: %s" % (y_test[i], p_test[i]));

