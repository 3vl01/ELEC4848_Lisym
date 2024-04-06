import sys
import glob

import numpy as np
import time
from msclap import CLAP

from keras.models import load_model




def main(argv):
    print(time.strftime("%H%M%S", time.localtime()))
    assert argv, 'Usage: testing.py <path>'
    classes = ['background','burning','coughing','gasStove','glass_breaking','plasticCollapse','runningTapWater','sneezing','sniffingNose']
    for c in classes:
        for file_name in glob.glob(argv[1]+'/'+c+'/*'):
            print(file_name)
            
            ground_truth = [c]
            audio_files = [file_name]

            # Load and initialize CLAP
            # Setting use_cuda = True will load the model on a GPU using CUDA
            clap_model = CLAP(version = '2023', use_cuda=False)

            # compute the audio embeddings from an audio file
            audio_embeddings = clap_model.get_audio_embeddings(audio_files, resample=True)
            em = np.asarray(audio_embeddings[0])

            my_model = load_model('final_BayesianOptimizationCnn1d_clap.h5') # change the file name to the one you want to test
            similarity = my_model(em.reshape(1, -1)).numpy()
            prediction = np.mean(similarity, axis=0)

            # Print the results
            print("Ground Truth: {}".format(ground_truth))
            print('Current event:\n' +
                    '\n'.join('  {:12s}: {:.3f}'.format(classes[i], prediction[i])
                                for i in range(9)))
    print(time.strftime("%H%M%S", time.localtime()))
    


if __name__ == '__main__':
    main(sys.argv)
