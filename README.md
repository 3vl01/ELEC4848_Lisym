# ELEC4848 A solution for Home Safety
The proposed solution is a sound-based home safety monitoring system targeting elderly. The system is called Lisym, Listen to symptoms.


## Authors

- [@3vl01](https://github.com/3vl01)

## Installation

Install important open-source library

```bash
  pip install msclap
```

## Running Tests

To run train your own sound classification model:
  1. Download [training dataset](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/ywl11_connect_hku_hk/EeLKRzTQyHJGqKIPIndKApcBNFKYe0g-QoPkTv3PM_NpDQ?e=ZFDboe)

  2. Run hyperparameter optimization and train it!

```bash
  cd TrainingModel
  // If it is based on YAMNet
  cd YAMNet
  python3 bestPara_Yamnet.py <path_to_trainingSet> <mode> // mode: r for random search; ob for Bayesian opt.
  python3 train_new.py <path_to_trainingSet> // Remember to change the model file name for saving and classify function;
  // Change the parameters according to the results of bestPara_Yamnet.py; Prepare a test wav file and change the path in classify function
  // If it is based on CLAP
  cd CLAP
  python3 bestPara_CLAP.py <path_to_trainingSet> <mode> // mode: r for random search; ob for Bayesian opt.
  python3 train_clap.py <path_to_trainingSet> // Remember to change the model file name for saving and classify function;
  // Change the parameters according to the results of bestPara_CLAP.py; Prepare a test wav file and change the path in classify function
```
To test models:
  1. Download [testing dataset](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/ywl11_connect_hku_hk/EQPg1vHfX2BFq6YiYFTF1iQB_AekRZd6UPtExS_Yeu7A5g?e=PG7qKC)
  2. (Optional) Download other models you want to try
     - [CLAP Minimized model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/ywl11_connect_hku_hk/EberdBRBqHFBprT5hlwS-2UBv-G04YzihKh3jyS8ebz57w?e=zoouv9)
     - [CLAP Random Search model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/ywl11_connect_hku_hk/EY0fCavvhntEknSwZvxFO6kB1ahDKdkjfR-c2FoUlqlTDA?e=Eqkg3K)
     - [CLAP Bayesian Optimization model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/ywl11_connect_hku_hk/EXPEuPu6-gBOhWYAM85vK94Bd3MHCS_kE1xPkHjtEVdHnQ?e=J7ZqMZ)
  3. Run the code below after change the model file name inside:
```bash
  cd Testing
  python3 testing.py <path_to_testingSet>
```

## Run on Raspberry Pi

To start the system on Raspberry Pi, move the folder PiApp to your Raspberry Pi and run the code below:
```bash
  cd PiApp
  python3 LisymPi.py
```

## About mobile app

Download the apk to run the app, only Android users can run it at this moment.
