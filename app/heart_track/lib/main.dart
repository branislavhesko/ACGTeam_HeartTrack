import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:path_provider/path_provider.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'package:device_info_plus/device_info_plus.dart';
import 'dart:async';
import 'dart:io';
import 'dart:convert';
import 'package:fl_chart/fl_chart.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;

  const MyApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: "ACGTeam's Heart Tracker",
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.red),
        useMaterial3: true,
      ),
      home: MyHomePage(title: "ACGTeam's Heart Tracker <3", cameras: cameras),
    );
  }
}

class MyHomePage extends StatefulWidget {
  final String title;
  final List<CameraDescription> cameras;

  const MyHomePage({super.key, required this.title, required this.cameras});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  CameraController? _cameraController;
  bool _isCameraInitialized = false;
  bool _isRecording = false;
  bool _isFlashOn = false;
  String? _uniqueID;
  Timer? _durationTimer;
  Timer? _dataFetchTimer;
  bool _dataFetchStarted = false;
  int _recordingDuration = 0;
  int _chunksUploaded = 0;
  List<double> rawPpg = [];
  List<double> processedPpg = [];
  String heartRate = '0';
  String okSignal = '0.0';
  double quality = 0.0;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _generateUniqueID();
  }

  Future<void> _initializeCamera() async {
    _cameraController = CameraController(
      widget.cameras.first,
      ResolutionPreset.low,
      enableAudio: true,
    );

    try {
      await _cameraController!.initialize();
      setState(() {
        _isCameraInitialized = true;
      });
    } catch (e) {
      print("Error initializing camera: $e");
    }
  }

  Future<void> _requestPermissions() async {
    var storageStatus = await Permission.storage.request();
    var cameraStatus = await Permission.camera.request();

    if (storageStatus.isGranted && cameraStatus.isGranted) {
      print("All permissions granted");
    } else {
      print("Permissions denied");
    }
  }

  Future<void> _generateUniqueID() async {
    final deviceInfo = DeviceInfoPlugin();
    String id;

    if (Platform.isAndroid) {
      var androidInfo = await deviceInfo.androidInfo;
      id = androidInfo.id ?? "unknown";
    } else if (Platform.isIOS) {
      var iosInfo = await deviceInfo.iosInfo;
      id = iosInfo.identifierForVendor ?? "unknown";
    } else {
      id = "unknown";
    }

    setState(() {
      _uniqueID = id;
    });

    print("Generated unique ID: $_uniqueID");
  }

  Future<void> _fetchData() async {
    if (_uniqueID == null) return;

    try {
      final response = await http.get(
        Uri.parse('http://152.53.50.74:8000/results?patient_id=$_uniqueID'),
        headers: {'accept': 'application/json'},
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          rawPpg = List<double>.from(data['raw_ppg']);
          processedPpg = List<double>.from(data['processed_ppg']);
          heartRate = data['heart_rate'];
          okSignal = data['OK signal'];
          quality = data['quality'];
        });
      } else {
        print("Failed to fetch data: ${response.statusCode}");
      }
    } catch (e) {
      print("Error fetching data: $e");
    }
  }

  void _startDataFetchTimer() {
    if (_dataFetchStarted) return;
    _dataFetchStarted = true;

    _dataFetchTimer = Timer.periodic(Duration(seconds: 1), (timer) {
      _fetchData();
    });
  }

  void _toggleFlashlight() async {
    if (_cameraController != null) {
      try {
        _isFlashOn = !_isFlashOn;
        await _cameraController!.setFlashMode(
          _isFlashOn ? FlashMode.torch : FlashMode.off,
        );
        setState(() {});
      } catch (e) {
        print("Error toggling flashlight: $e");
      }
    }
  }

  void _toggleRecording() {
    setState(() {
      _isRecording = !_isRecording;
    });

    if (_isRecording) {
      _startRecordingLoop();
    } else {
      _stopRecording();
    }
  }

  Future<void> _startRecordingLoop() async {
    setState(() {
      _recordingDuration = 0;
      _chunksUploaded = 0;
    });

    _durationTimer = Timer.periodic(Duration(seconds: 1), (timer) {
      setState(() {
        _recordingDuration++;
      });
    });

    // Simulating continuous recording and uploading loop
    while (_isRecording) {
      await _recordAndUploadChunk();
    }
  }

  Future<void> _recordAndUploadChunk() async {
    if (_uniqueID == null || !_isCameraInitialized) return;

    try {
      Directory appDocDir = await getApplicationDocumentsDirectory();
      final videoFilePath =
          '${appDocDir.path}/video_record_${DateTime.now().millisecondsSinceEpoch}.mp4';

      // Start recording
      await _cameraController!.startVideoRecording();
      print("Recording started...");

      // Wait for 10 seconds to record a chunk
      await Future.delayed(Duration(seconds: 10));

      // Stop recording
      final videoFile = await _cameraController!.stopVideoRecording();
      print("Recording stopped. File saved at: ${videoFile.path}");

      // Upload video chunk
      await _uploadFile(videoFile.path);
      setState(() {
        _chunksUploaded++;
      });

    } catch (e) {
      print("Error during recording or uploading: $e");
      _stopRecording();
    }
  }

  Future<void> _uploadFile(String videoFilePath) async {
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('http://152.53.50.74:8000/upload_video?device_id=$_uniqueID'),
      );
      request.files.add(await http.MultipartFile.fromPath('video', videoFilePath));
      var response = await request.send();

      if (response.statusCode == 200) {
        print("Video chunk uploaded successfully");
        if (!_dataFetchStarted) {
          _startDataFetchTimer();
        }
      } else {
        print("Failed to upload video chunk: ${response.statusCode}");
      }
    } catch (e) {
      print("Error uploading video file: $e");
    }
  }

  void _stopRecording() {
    setState(() {
      _isRecording = false;
    });

    _durationTimer?.cancel();
    _durationTimer = null;
  }

  List<FlSpot> _getPpgData(List<double> ppgData) {
    return List.generate(
      ppgData.length,
      (index) => FlSpot(index.toDouble(), ppgData[index]),
    );
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _durationTimer?.cancel();
    _dataFetchTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: _isCameraInitialized
          ? Column(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                AspectRatio(
                  aspectRatio: _cameraController!.value.aspectRatio,
                  child: CameraPreview(_cameraController!),
                ),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    ElevatedButton(
                      onPressed: _toggleRecording,
                      child: Text(_isRecording ? "Stop" : "Start"),
                    ),
                    const SizedBox(width: 20),
                    IconButton(
                      icon: Icon(
                        _isFlashOn ? Icons.flash_on : Icons.flash_off,
                      ),
                      onPressed: _toggleFlashlight,
                    ),
                  ],
                ),
                Text(
                  "Duration: $_recordingDuration s | Chunks: $_chunksUploaded",
                  style: TextStyle(fontSize: 14),
                ),
                Text(
                  "Heart Rate: $heartRate BPM | OK Signal: $okSignal | Quality: $quality",
                  style: TextStyle(fontSize: 14),
                ),
                SizedBox(
                  height: 100,
                  child: LineChart(
                    LineChartData(
                      lineBarsData: [
                        LineChartBarData(
                          spots: _getPpgData(rawPpg),
                          isCurved: true,
                          color: Colors.red,
                          dotData: FlDotData(show: false),
                        ),
                      ],
                      titlesData: FlTitlesData(show: false),
                      borderData: FlBorderData(show: false),
                      gridData: FlGridData(show: false),
                    ),
                  ),
                ),
                SizedBox(
                  height: 100,
                  child: LineChart(
                    LineChartData(
                      lineBarsData: [
                        LineChartBarData(
                          spots: _getPpgData(processedPpg),
                          isCurved: true,
                          color: Colors.green,
                          dotData: FlDotData(show: false),
                        ),
                      ],
                      titlesData: FlTitlesData(show: false),
                      borderData: FlBorderData(show: false),
                      gridData: FlGridData(show: false),
                    ),
                  ),
                ),
              ],
            )
          : Center(child: CircularProgressIndicator()),
    );
  }
}
