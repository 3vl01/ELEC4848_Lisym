import time
print("Start program: "+time.strftime("%H%M%S", time.localtime()))
import os
import numpy as np
from keras.models import load_model
from msclap import CLAP
import sounddevice as sd
import soundfile as sf
import asyncio
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import firestore


# Parameters
duration = 2 # Duration in seconds
sample_rate = 44100 # Sample rate (Hz)
channels = 1 # Mono audio
max_queue_size = 1 # Maximum number of audio files in the queue


cred = credentials.Certificate("fypfirebase-389cb-firebase-adminsdk-zqohn-77b09d2204.json")
firebase_admin.initialize_app(cred, {
  'databaseURL': 'https://fypfirebase-389cb-default-rtdb.firebaseio.com/'
})
ref = db.reference('Tryfirebase/')
class_ref = ref.child('class')


classes = ['background','burning','coughing','gasStove','glass_breaking','plasticCollapse','runningTapWater','sneezing','sniffingNose']
clap_model = CLAP(version = '2023', use_cuda=False)
my_model = load_model('final_MinimumCnn1d_clap.h5')



async def classify(queue):
  while True:
    # Get an audio file from the queue
    filename = await queue.get()
    print("Time of recording: "+filename)
    print("Start classify: "+time.strftime("%H%M%S", time.localtime()))

    # Get the audio embedding using CLAP
    audio_embeddings = clap_model.get_audio_embeddings([filename], resample=True)
    em = np.asarray(audio_embeddings[0])

    # Get the similarity scores using the custom model
    similarity = my_model(em.reshape(1, -1)).numpy()
    prediction = np.mean(similarity, axis=0)

    print("Finish classification: "+time.strftime("%H%M%S", time.localtime()))
    # Print the results
    i = np.argmax(prediction)
    if i != 0:
      each_ref = class_ref.child(classes[i])
      date = time.strftime("%y%m%d", time.localtime())
      each_ref.child(date).child(filename[7:-4]).set(filename[7:-4])

    print("Finish upload: "+time.strftime("%H%M%S", time.localtime()))
    print('Current event:\n' + '\n'.join(' {:12s}: {:.3f}'.format(classes[i], prediction[i]) for i in range(9)))
    # Delete the audio file
    os.remove(filename)
    # Mark the queue task as done
    queue.task_done()

async def main():
  cnt = 0
  # Create a queue to store the audio files
  queue = asyncio.Queue(maxsize=max_queue_size)
  # Create a coroutine to classify the audio files
  classifier = asyncio.create_task(classify(queue))

  while True:
    # Record audio
    t = time.localtime()
    current_time = time.strftime("%y%m%d_%H%M%S", t)
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
    print(current_time)
    sd.wait() # Wait for the recording to complete

    # Save audio to WAV file
    output_filename = str(current_time)+".wav" # Output WAV file name
    sf.write(output_filename, recording, sample_rate)
    await queue.put(output_filename)
    cnt += 1

asyncio.run(main())