# Federated-Keystroke-Detection
## Data
We use the dataset created by [Dhakal et al.](https://userinterfaces.aalto.fi/136Mkeystrokes/)

Data in the dataset has the following form:

|PARTICIPANT_ID	| TEST_SECTION_ID |	SENTENCE |	USER_INPUT | KEYSTROKE_ID |	PRESS_TIME	| RELEASE_TIME |	LETTER	| KEYCODE|
|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
|100007	| 1091059 |	But the pact has not entirely calmed the nation's nerves.| 	But the pact has not entirely calmed the nation's nerves. 	| 51895219 |	1473275416530 |	1473275416839	| SHIFT |	16 |
|100007	| 1091059 |	But the pact has not entirely calmed the nation's nerves.|	But the pact has not entirely calmed the nation's nerves. |	51895226 |	1473275416749 |	1473275416886 |	B |	66|<br>
| ... |

Please see the data directory for a few sample files.

### Data Preprocessing
To generate the human data for the federated learning, run the [model/data_reader.py](model/data_reader.py) script. This script reads the raw data from the csv files and generates the data that is used by the model. The raw files are not included in the repository, please download them and place them in [Keystrokes/files/](Keystrokes/files/) in top level of the repository. The pre-processed data is stored in the [data](data) directory.

To generate the bot data, run [keystroke_generator/bot_data.py](keystroke_generator/bot_data.py). The script will generate 16 dataset files with in total about 108,000,000 keystrokes to pretrain the model with. The bot datasets are also placed in the [data](data) directory.

## Web UI
The web frontend is located in [web_app/federated_keystroke_detection](./web_app/federated_keystroke_detection). It is a flutter application. The actual code is located
in [main.dart](web_app/federated_keystroke_detection/lib/main.dart).

### Code
The Application is a simple screen with a textfield and buttons for different functionalities. These are described into depth in the report. The screen itself is build
in the ``build()`` method of the ``MyHomePage`` class.

The other methods are used for building the different objects (such as buttons, or textfields) or for functionalities as converting a keypress on the keyboard into a data
keypress.

### Developement
To change things in the code and test locally. [Flutter has to be installed](https://docs.flutter.dev/get-started/install).
To build and host the frontend the command ``flutter build web`` can be used. This builds the web app. The files can be found in [build/web](web_app/federated_keystroke_detection/build/web). This can
then be hosted in Apache2 or similar applications.

## Model
Train the model by calling the [model/train.py](model/train.py) script. 
The script will train the model on the human data and then evaluate the model on the human and bot data.

## Oakestra setup
We use Oakestra for the federated learning. To run the project on Oakestra, follow the instructions in by [oakestra](https://github.com/oakestra/oakestra).
Stick to the official oakestra setup guide, but add our images to the config, and open port 80.
To install the images run
```docker pull ghcr.io/27robert99/master_node:latest```

