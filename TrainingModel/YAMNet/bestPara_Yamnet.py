import sys
import os
import numpy as np
import librosa
import tensorflow as tf

import keras_tuner
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
import yamnet
from yamnet import yamnet_frames_model
from params import Params
import matplotlib.pyplot as plt


YAMNET_PATH = "./yamnet.h5"


def create_dataset(path):
    samples, labels = [], []
    train_samples = []
    train_label = []
    val_samples = []
    val_label = []
    train_soundpath = []
    val_soundpath = []
    soundpath = []
    model = yamnet_frames_model(Params())
    model.load_weights(YAMNET_PATH)
    for cls in os.listdir(path):
        cnt = 0
        for sound in tqdm(os.listdir(os.path.join(path, cls))):
            if cls == "background":
              label = 0
            elif cls == "burning":
              label = 1
            elif cls == "coughing":
              label = 2
            elif cls == "gasStove":
              label = 3
            elif cls == "glass_breaking":
              label = 4
            elif cls == "plasticCollapse":
              label = 5
            elif cls == "runningTapWater":
              label = 6
            elif cls == "sneezing":
              label = 7
            elif cls == "sniffingNose":
              label = 8
            wav = librosa.load(os.path.join(os.path.join(path, cls, sound)), sr=16000)[0].astype(np.float32)


            for feature in model(wav)[1]:
                if cnt < 48:
                    train_samples.append(feature)
                    train_label.append(label)
                else:
                    val_samples.append(feature)
                    val_label.append(label)
            cnt += 1
    train_samples = np.asarray(train_samples)
    train_samples = train_samples.astype(np.float32)
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
    assert argv, 'Usage: bestPara_Yamnet.py <path_to_data> <mode>'
    path = argv[0]
    mode = argv[1]
    train_s, train_l, val_s, val_l = create_dataset(path)
    model = train_model(train_s, train_l, val_s, val_l, mode)


if __name__ == "__main__":
    main(sys.argv[1:])
