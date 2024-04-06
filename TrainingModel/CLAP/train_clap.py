import sys
import os
from tqdm import tqdm
from msclap import CLAP
import torch.nn.functional as F
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def create_dataset(path):
    samples, labels = [], []
    train_samples = []
    train_label = []
    val_samples = []
    val_label = []
    train_soundpath = []
    val_soundpath = []
    soundpath = []
    clap_model = CLAP(version = '2023', use_cuda=False)
    for cls in os.listdir(path):
        for sound in tqdm(os.listdir(os.path.join(path, cls))):
            soundpath.append(path +"/"+ cls + "/" + sound)
            if cls == "background":
              labels.append(0)
            elif cls == "burning":
              labels.append(1)
            elif cls == "coughing":
              labels.append(2)
            elif cls == "gasStove":
              labels.append(3)
            elif cls == "glass_breaking":
              labels.append(4)
            elif cls == "plasticCollapse":
              labels.append(5)
            elif cls == "runningTapWater":
              labels.append(6)
            elif cls == "sneezing":
              labels.append(7)
            elif cls == "sniffingNose":
              labels.append(8)
        for i in range(48):
          train_soundpath.append(soundpath[i])
          train_label.append(labels[i])
        for i in range(48, 60):
          val_soundpath.append(soundpath[i])
          val_label.append(labels[i])
        labels = []
        soundpath = []
    train_samples = clap_model.get_audio_embeddings(train_soundpath, resample=True)
    train_samples = np.asarray(train_samples)
    train_samples = train_samples.astype(np.float32)
    val_samples = clap_model.get_audio_embeddings(val_soundpath, resample=True)
    val_samples = np.asarray(val_samples)
    val_samples = val_samples.astype(np.float32)
    train_label = np.asarray(train_label)
    val_label = np.asarray(val_label)
    return train_samples, train_label, val_samples, val_label

def train_model(train_samples, train_label, val_samples, val_label):

  my_classes = 9
  # hyperparameter with best performance
  my_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(1024, 1)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(my_classes, activation='softmax')
  ])
  

  my_model.summary()

  my_model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),      
      optimizer="adam",
      metrics=['accuracy']
  )

  callback = tf.keras.callbacks.EarlyStopping(
      monitor='loss',
      patience=5,
      restore_best_weights=True
  )
  
  history = my_model.fit(
      train_samples,
      train_label,
      epochs=50,
      validation_data=(val_samples, val_label),
      callbacks=callback
  )
  

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(acc, label='Training Accuracy')
  plt.plot(val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.ylabel('Accuracy')
  plt.ylim([min(plt.ylim()),1])
  plt.title('Training and Validation Accuracy (CLAP CNN 1D Minimum)')

  plt.subplot(2, 1, 2)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.ylabel('Cross Entropy')
  plt.ylim([0,1.0])
  plt.title('Training and Validation Loss (CLAP CNN 1D Minimum)')
  plt.xlabel('epoch')
  plt.show()

  my_model.save('final_minimumCnn1d_clap.h5') # save your model with a new name
  print("===CNN Minimum model by transfer learning on CLAP===")
  print("Average train accuracy: " + str(np.mean(acc)))
  print("Average train loss: " + str(np.mean(loss)))
  print("Max train accuracy: " + str(np.max(acc)))
  print("Min train loss: " + str(np.min(loss)))
  print("Average val accuracy: " + str(np.mean(val_acc)))
  print("Average val loss: " + str(np.mean(val_loss)))
  print("Max val accuracy: " + str(np.max(val_acc)))
  print("Min val loss: " + str(np.min(val_loss)))

def classify():
  # Define classes for zero-shot
  # Should be in lower case and can be more than one word
  classes = ['background','burning','coughing','gasStove','glass_breaking','plasticCollapse','runningTapWater','sneezing','sniffingNose']
  ground_truth = ['sneezing']
  # Add prompt
  prompt = 'this is a sound of '
  class_prompts = [prompt + x for x in classes]
  #Load audio files
  audio_files = ['./audio_test/sneeze1.wav']

  # Load and initialize CLAP
  # Setting use_cuda = True will load the model on a GPU using CUDA
  clap_model = CLAP(version = '2023', use_cuda=False)

  # compute text embeddings from natural text
  text_embeddings = clap_model.get_text_embeddings(class_prompts)

  # compute the audio embeddings from an audio file
  audio_embeddings = clap_model.get_audio_embeddings(audio_files, resample=True)
  em = np.asarray(audio_embeddings[0])


  my_model = load_model('final_minimumCnn1d_clap.h5') #change the file name
  similarity = my_model(em.reshape(1, -1)).numpy()
  prediction = np.mean(similarity, axis=0)

  # Print the results
  print("Ground Truth: {}".format(ground_truth))
  print('Current event:\n' +
          '\n'.join('  {:12s}: {:.3f}'.format(classes[i], prediction[i])
                    for i in range(9)))

def main(argv):
    assert argv, 'Usage: train_clap.py <path_to_data>'
    path = argv[0]
    train_s, train_l, val_s, val_l = create_dataset(path)
    model = train_model(train_s, train_l, val_s, val_l)
    classify()


if __name__ == "__main__":
    main(sys.argv[1:])