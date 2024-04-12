import 'dart:convert';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Keystroke Classification as CAPTCHA method',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal)
            .copyWith(background: Colors.grey.shade200),
      ),
      home:
          const MyHomePage(title: 'Keystroke Classification as CAPTCHA method'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  //Tracking keystrokes
  late final FocusNode _focusNode;
  final fieldText = TextEditingController();

  //Colors of the classification fields
  Color _humanColor = Colors.white;
  Color _botColor = Colors.white;

  //Classification text
  String human_classification = "";
  String bot_classification = "";

  // Keystroke data
  int id = 0;
  DateTime? _keyDownTime;
  int? _duration;
  int? _jsKeyCode;
  int? _testSectionId;
  String _sentence = "";

  //temporary keystrokes
  Map<String, Map<String, int>> _keystrokesInSentence = {};
  int _keystrokeCounter = 0;
  Map<String, Map<String, Map<String, int>>> _keyStrokesInMemory = {};

  void _prepareVariablesForNewKeystrokes() {
    _testSectionId = Random().nextInt(9999999);
    _keystrokesInSentence = {};
    _keystrokeCounter = 0;
    _sentence = "";
    fieldText.clear();
  }

  @override
  void initState() {
    super.initState();
    _prepareVariablesForNewKeystrokes();
    _focusNode = FocusNode();
  }

  @override
  void dispose() {
    _focusNode.dispose();
    super.dispose();
  }

  void _handleKeyDownEvent(RawKeyEvent event) {
    _jsKeyCode = event.data.isShiftPressed
        ? 16
        : flutterKeyToJsKeyCode(event.data.logicalKey.keyId);
    _keyDownTime = DateTime.now();
    _sentence += event.data.logicalKey.keyLabel;
  }

  void _handleKeyUpEvent(RawKeyEvent event) {
    _duration = DateTime.now().difference(_keyDownTime!).inMilliseconds;
    _keystrokesInSentence[_keystrokeCounter.toString()] = {
      "pressTime": _keyDownTime!.millisecondsSinceEpoch,
      "duration": _duration!,
      "testSectionId": _testSectionId!,
      "jsKeyCode": _jsKeyCode!,
    };
    _keystrokeCounter++;
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
          backgroundColor: Theme.of(context).colorScheme.inversePrimary,
          title: Text(widget.title)),
      body: Container(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            _textEntryField(),
            const SizedBox(height: 10),
            _classificationViews(),
            const SizedBox(height: 20),
            _saveDataView(),
            const SizedBox(height: 10),
            _sendToBackendView(),
          ],
        ),
      ),
    );
  }

  Container _textEntryField() {
    return Container(
      padding: EdgeInsets.symmetric(vertical: 10, horizontal: 20),
      decoration: BoxDecoration(
          color: Colors.white,
          border: Border.all(color: Colors.grey),
          borderRadius: BorderRadius.circular(25)),
      child: RawKeyboardListener(
        autofocus: false,
        focusNode: _focusNode,
        onKey: (event) {
          if (event is RawKeyDownEvent) {
            _handleKeyDownEvent(event);
          } else if (event is RawKeyUpEvent) {
            _handleKeyUpEvent(event);
          }
        },
        child: TextField(
          onTap: () => setState(() {
            _humanColor = Colors.white;
            _botColor = Colors.white;
            human_classification = "";
            bot_classification = "";
          }),
          decoration: InputDecoration(hintText: "Sentence to classify or save"),
          controller: fieldText,
        ),
      ),
    );
  }

  Row _classificationViews() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        _classificationView("Human Keystrokes",
            _sendKeystrokesToServerForHumanClassification, "classification"),
        _classificationView("Bot Keystrokes",
            _sendKeystrokesToServerForBotClassification, "botclassification"),
      ],
    );
  }

  Expanded _classificationView(
      String text, Function() classificationString, String functionString) {
    return Expanded(
      child: Container(
        decoration: BoxDecoration(
            color:
                (functionString == "classification") ? _humanColor : _botColor,
            border: Border.all(color: Colors.grey),
            borderRadius: BorderRadius.circular(25)),
        margin: EdgeInsets.all(40),
        padding: EdgeInsets.symmetric(vertical: 20),
        child: Column(children: [
          Text(text),
          SizedBox(height: 10),
          Text(style: TextStyle(color: Colors.grey), fieldText.text),
          SizedBox(height: 20),
          Text((functionString == "classification")
              ? human_classification
              : bot_classification),
          Divider(),
          SizedBox(height: 20),
          ElevatedButton(
              onPressed: () async {
                String classification = await classificationString();
                setState(() {
                  if (functionString == "classification") {
                    if (classification == "Pass") {
                      human_classification = "Classified as human keystrokes";
                      _humanColor = Colors.green.shade700;
                    } else if (classification == "Fail") {
                      human_classification = "Classified as bot keystrokes";
                      _humanColor = Colors.red.shade700;
                    } else {
                      human_classification = "There was an error!";
                      _humanColor = Colors.red.shade700;
                    }
                  } else {
                    if (classification == "Pass") {
                      bot_classification = "Classified as bot keystrokes";
                      _botColor = Colors.green.shade700;
                    } else if (classification == "Fail") {
                      bot_classification = "Classified as human keystrokes";
                      _botColor = Colors.red.shade700;
                    } else {
                      bot_classification = "There was an error!";
                      _botColor = Colors.red.shade700;
                    }
                  }
                });
              },
              child: Text("Classify")),
        ]),
      ),
    );
  }

  Container _saveDataView() {
    return Container(
      padding: EdgeInsets.symmetric(vertical: 15),
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border.all(color: Colors.grey),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          Text("Save data for local training"),
          ElevatedButton(
              onPressed: _createKeystrokesAndSaveLocally, child: Text("Save"))
        ],
      ),
    );
  }

  ElevatedButton _sendToBackendView() {
    return ElevatedButton(
        onPressed: _sendToServerForTraining,
        child: Text("Send to Backend for local training"));
  }

  void _createKeystrokesAndSaveLocally() {
    _keyStrokesInMemory[id.toString()] = _keystrokesInSentence;
    id++;
    _prepareVariablesForNewKeystrokes();
  }

  Future<void> _sendToServerForTraining() async {
    id = 0;
    Map<String, String> headers = {
      'Content-Type': 'application/json',
    };
    try {
      http.Response response = await http.post(
        Uri.parse('http://localhost:8080/training'),
        headers: headers,
        body: json.encode(_keyStrokesInMemory),
      );
    } catch (e) {
      print("Exception caught: $e");
    }
    _keyStrokesInMemory = {};
  }

  Future<String> _sendKeystrokesToServerForHumanClassification() async {
    Map<String, String> headers = {
      'Content-Type': 'application/json',
    };
    String result = "";
    try {
      http.Response response = await http.post(
        Uri.parse('http://localhost:8080/classification'),
        headers: headers,
        body: json.encode(_keystrokesInSentence),
      );
      if (response.statusCode == 200) {
        var responseData = response.body;
        result = responseData;
      } else {
        result = response.statusCode as String;
      }
    } catch (e) {
      print(e);
      result = e.toString();
    }
    _prepareVariablesForNewKeystrokes();
    return result;
  }

  Future<String> _sendKeystrokesToServerForBotClassification() async {
    Map<String, String> headers = {
      'Content-Type': 'text/html',
    };
    String result = "";
    try {
      http.Response response = await http.post(
        Uri.parse('http://localhost:8080/botclassification'),
        headers: headers,
        body: fieldText.text,
      );
      if (response.statusCode == 200) {
        var responseData = response.body;
        result = responseData;
      } else {
        result = response.statusCode as String;
      }
    } catch (e) {
      print(e);
      result = e.toString();
    }
    _prepareVariablesForNewKeystrokes();
    return result;
  }

  int? flutterKeyToJsKeyCode(int flutterKeyCode) {
    var mapping = <int, int>{
      // Letters
      LogicalKeyboardKey.keyA.keyId: 65,
      // 'A'
      LogicalKeyboardKey.keyB.keyId: 66,
      // 'B'
      LogicalKeyboardKey.keyC.keyId: 67,
      // 'C'
      LogicalKeyboardKey.keyD.keyId: 68,
      // 'A'
      LogicalKeyboardKey.keyE.keyId: 69,
      // 'B'
      LogicalKeyboardKey.keyF.keyId: 70,
      // 'C'
      LogicalKeyboardKey.keyG.keyId: 71,
      // 'A'
      LogicalKeyboardKey.keyH.keyId: 72,
      // 'B'
      LogicalKeyboardKey.keyI.keyId: 73,
      // 'C'
      LogicalKeyboardKey.keyJ.keyId: 74,
      // 'A'
      LogicalKeyboardKey.keyK.keyId: 75,
      // 'C'
      LogicalKeyboardKey.keyL.keyId: 76,
      // 'A'
      LogicalKeyboardKey.keyM.keyId: 77,
      // 'B'
      LogicalKeyboardKey.keyN.keyId: 78,
      // 'C'
      LogicalKeyboardKey.keyO.keyId: 79,
      // 'A'
      LogicalKeyboardKey.keyP.keyId: 80,
      // 'B'
      LogicalKeyboardKey.keyQ.keyId: 81,
      // 'A'
      LogicalKeyboardKey.keyR.keyId: 82,
      // 'B'
      LogicalKeyboardKey.keyS.keyId: 83,
      // 'C'
      LogicalKeyboardKey.keyT.keyId: 84,
      // 'A'
      LogicalKeyboardKey.keyU.keyId: 85,
      // 'C'
      LogicalKeyboardKey.keyV.keyId: 86,
      // 'A'
      LogicalKeyboardKey.keyW.keyId: 87,
      // 'B'
      LogicalKeyboardKey.keyX.keyId: 88,
      // 'C'
      LogicalKeyboardKey.keyY.keyId: 89,
      // 'A'
      LogicalKeyboardKey.keyZ.keyId: 90,
      // 'B'

      // Numbers (top row)
      LogicalKeyboardKey.digit0.keyId: 48,
      // '1'
      LogicalKeyboardKey.digit1.keyId: 49,
      // '1'
      LogicalKeyboardKey.digit2.keyId: 50,
      // '2'
      LogicalKeyboardKey.digit3.keyId: 51,
      // '3'
      LogicalKeyboardKey.digit4.keyId: 52,
      // '1'
      LogicalKeyboardKey.digit5.keyId: 53,
      // '2'
      LogicalKeyboardKey.digit6.keyId: 54,
      // '3'
      LogicalKeyboardKey.digit7.keyId: 55,
      // '1'
      LogicalKeyboardKey.digit8.keyId: 56,
      // '2'
      LogicalKeyboardKey.digit9.keyId: 57,
      // '3'

      // Control keys
      LogicalKeyboardKey.enter.keyId: 13,
      // Enter
      LogicalKeyboardKey.space.keyId: 32,
      // Space
      LogicalKeyboardKey.backspace.keyId: 8,
      // Backspace
      LogicalKeyboardKey.escape.keyId: 27,
      // Escape
      LogicalKeyboardKey.shiftLeft.keyId: 16,
      // Left Shift
      LogicalKeyboardKey.shiftRight.keyId: 16,
      // Right Shift (Note: JS does not distinguish left/right)
      LogicalKeyboardKey.controlLeft.keyId: 17,
      // Left Control
      LogicalKeyboardKey.controlRight.keyId: 17,
      // Right Control
      LogicalKeyboardKey.altLeft.keyId: 18,
      // Left Alt
      LogicalKeyboardKey.altRight.keyId: 18,
      // Right Alt
      LogicalKeyboardKey.arrowUp.keyId: 38,
      // Arrow Up
      LogicalKeyboardKey.arrowDown.keyId: 40,
      // Arrow Down
      LogicalKeyboardKey.arrowLeft.keyId: 37,
      // Arrow Left
      LogicalKeyboardKey.arrowRight.keyId: 39,
      // Arrow Right
    };
    return mapping[flutterKeyCode] ?? 16;
  }
}
