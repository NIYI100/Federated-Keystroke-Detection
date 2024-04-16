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