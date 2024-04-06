import sys
import os
from tqdm import tqdm
from msclap import CLAP
import torch.nn.functional as F
import keras_tuner
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
        for i in range(54):
          train_soundpath.append(soundpath[i])
          train_label.append(labels[i])
        for i in range(54, 60):
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

def build_model(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(hp.Int('conv1_filters', 32, 128, step=32), 3, activation='relu', input_shape=(1024, 1)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(hp.Int('conv2_filters', 64, 256, step=32), 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(hp.Int('conv3_filters', 64, 256, step=32), 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hp.Int('dense_units', 256, 1024, step=128), activation='relu'),
        tf.keras.layers.Dense(9, activation='softmax')
    ])
  

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(train_samples, train_label, val_samples, val_label, mode):
    # Define the tuner
    if mode == 'r':
        tuner = keras_tuner.RandomSearch(
            build_model,
            objective='val_accuracy',
            max_trials=30,
            executions_per_trial=2,
            directory='my_tuner_dir_ran',
            project_name='sound_classification_tuner_ran'
        )
    else:
        tuner = keras_tuner.BayesianOptimization(
            build_model,
            objective='val_accuracy',
            max_trials=30,
            directory='my_tuner_dir_bo',
            project_name='sound_classification_tuner_bo'
        )

    # Perform hyperparameter tuning
    tuner.search(train_samples, train_label, epochs=10, validation_data=(val_samples, val_label))

    # Get the best hyperparameter configuration
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Evaluate the best model on the test set
    test_loss, test_accuracy = best_model.evaluate(val_samples, val_label)
    print('Test accuracy:', test_accuracy)

    # Print the best hyperparameters
    print('Best Hyperparameters:')
    print(best_hyperparameters.get_config())

def main(argv):
    assert argv, 'Usage: bestPara_CLAP.py <path_to_data> <mode>' # r for random search method, bo for Bayesian opt. method
    path = argv[0]
    mode = argv[1]
    train_s, train_l, val_s, val_l = create_dataset(path)
    model = train_model(train_s, train_l, val_s, val_l, mode)


if __name__ == "__main__":
    main(sys.argv[1:])