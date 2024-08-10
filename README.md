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
  1. Create training dataset.
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
  1. Create testing dataset.
  2. Run the code below after change the model file name inside:
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

## Report, Technical Paper, Poster
1. [Report](https://connecthkuhk-my.sharepoint.com/:b:/g/personal/ywl11_connect_hku_hk/EbeolD0unTpNlgIoR8XxrD8BX8b67UR2ik4YtyOu1_-3Kg?e=pWvLMa)
2. [Technical Paper](https://connecthkuhk-my.sharepoint.com/:b:/g/personal/ywl11_connect_hku_hk/EekWDLz4qqNOju0QqbcEINYBDeUiMdklopLCUlzJ4CAAKQ?e=eyaZRC)
3. [Poster](https://connecthkuhk-my.sharepoint.com/:b:/g/personal/ywl11_connect_hku_hk/EcR_mL3DZi1Al01x53cc9qUBRxefoQTFc9tU2DuGzMgoGg?e=RLEl2h)
