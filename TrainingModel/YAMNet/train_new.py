import sys
import os
import numpy as np
import librosa
import resampy
import tensorflow as tf
import soundfile as sf

import keras_tuner
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from keras.models import load_model
import yamnet
from yamnet import yamnet_frames_model
from params import Params
import matplotlib.pyplot as plt
import params as yamnet_params
import yamnet as yamnet_model


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

def train_model(train_samples, train_label, val_samples, val_label):
    my_classes = 9
    # change the parameters
    my_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(1024, 1)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(224, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(640, activation='relu'),
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
    plt.title('Training and Validation Accuracy (YAMnet CNN 1D Bayesian Optimization)') # change with suitable title

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss (YAMnet CNN 1D Bayesian Optimization)') # change with suitable title
    plt.xlabel('epoch')
    plt.show()

    my_model.save('final_BayesianOptimizationCnn1d_yamnet.h5') # save with your new file name
    print("===CNN Bayesian Optimization model by transfer learning on YAMnet===") # change with suitable title
    print("Average train accuracy: " + str(np.mean(acc)))
    print("Average train loss: " + str(np.mean(loss)))
    print("Max train accuracy: " + str(np.max(acc)))
    print("Min train loss: " + str(np.min(loss)))
    print("Average val accuracy: " + str(np.mean(val_acc)))
    print("Average val loss: " + str(np.mean(val_loss)))
    print("Max val accuracy: " + str(np.max(val_acc)))
    print("Min val loss: " + str(np.min(val_loss)))

def classify():
    params = yamnet_params.Params()
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('yamnet.h5')
    yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')
    file_name = './audio_test/sneeze1.wav'
    # Decode the WAV file.
    wav_data, sr = sf.read(file_name, dtype=np.int16)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    waveform = waveform.astype('float32')

    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != params.sample_rate:
        waveform = resampy.resample(waveform, sr, params.sample_rate)

    # Predict YAMNet classes.
    scores, embeddings, spectrogram = yamnet(waveform)
    my_model = load_model('final_BayesianOptimizationCnn1d_yamnet.h5') # test with your new file name
    result = my_model(embeddings).numpy()

    confidence_values = result.mean(axis=0) * 100

    my_classes = ['background','burning','coughing','gasStove','glass_breaking','plasticCollapse','runningTapWater','sneezing','sniffingNose']

    inferred_class = my_classes[confidence_values.argmax()]
    print(f'The main sound is: {inferred_class} with a confidence of {confidence_values.max()}%')


def main(argv):
    assert argv, 'Usage: train_new.py <path_to_data>'
    path = argv[0]
    train_s, train_l, val_s, val_l = create_dataset(path)
    model = train_model(train_s, train_l, val_s, val_l)
    classify()


if __name__ == "__main__":
    main(sys.argv[1:])
