import 'dart:async';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:just_audio/just_audio.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'api_service.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Audio Analyzer',
      theme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.teal),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  static const String _defaultApiUrl =
      'https://keep-demodulation-or-use-demodulation-api.onrender.com';
  static const List<String> _supportedInputExtensions = [
    'wav',
    'mp3',
    'm4a',
    'flac',
    'ogg',
    'mp4',
    'mov',
    'mkv',
    'avi',
    'webm',
    'm4v',
  ];
  static const Set<String> _videoExtensions = {
    'mp4',
    'mov',
    'mkv',
    'avi',
    'webm',
    'm4v',
  };
  ApiService? _api;
  final AudioPlayer _audioPlayer = AudioPlayer();
  final AudioRecorder _audioRecorder = AudioRecorder();

  String? selectedPath;
  String? recordedPath;
  String? denoisedPath;
  String? _currentPlayingPath;
  bool _selectedInputIsVideo = false;
  String apiUrl = _defaultApiUrl;
  String resultText = 'Pick an audio/video file and run denoise + analysis.';
  String? predictedLabel;
  double? confidence;
  int? lastRequestMs;
  bool loading = false;
  bool _isRecording = false;
  int _recordingSeconds = 0;
  double _recordingLevel = 0.0;
  StreamSubscription<Amplitude>? _amplitudeSub;
  Timer? _recordingTicker;

  @override
  void initState() {
    super.initState();
    _loadSettings();
  }

  @override
  void dispose() {
    _amplitudeSub?.cancel();
    _recordingTicker?.cancel();
    _audioPlayer.dispose();
    _audioRecorder.dispose();
    super.dispose();
  }

  String _formatRecordingTime(int totalSeconds) {
    final minutes = (totalSeconds ~/ 60).toString().padLeft(2, '0');
    final seconds = (totalSeconds % 60).toString().padLeft(2, '0');
    return '$minutes:$seconds';
  }

  void _setSelectedInput(String path, {bool fromRecording = false}) {
    final isVideo = _isVideoPath(path);
    setState(() {
      selectedPath = path;
      _selectedInputIsVideo = isVideo;
      if (fromRecording) {
        recordedPath = path;
      }
      denoisedPath = null;
      predictedLabel = null;
      confidence = null;
      lastRequestMs = null;
      resultText = isVideo
          ? 'Video selected. Backend will extract audio and denoise it.'
          : 'Input ready. You can denoise or analyze now.';
    });
  }

  Future<void> _loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    final saved = prefs.getString('api_url');
    if (!mounted) {
      return;
    }
    setState(() {
      apiUrl = (saved == null || saved.trim().isEmpty) ? _defaultApiUrl : saved.trim();
      _api = ApiService(apiUrl);
    });
  }

  Future<void> _saveApiUrl(String value) async {
    final normalized = value.trim();
    if (normalized.isEmpty) {
      return;
    }
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('api_url', normalized);
    if (!mounted) {
      return;
    }
    setState(() {
      apiUrl = normalized;
      _api = ApiService(apiUrl);
    });
    _showToast('API URL saved.');
  }

  void _showToast(String message) {
    if (!mounted) {
      return;
    }
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text(message)));
  }

  Future<void> _playAudio(String path) async {
    try {
      await _audioPlayer.stop();
      await _audioPlayer.setFilePath(path);
      await _audioPlayer.play();
      if (!mounted) {
        return;
      }
      setState(() {
        _currentPlayingPath = path;
      });
    } catch (exc) {
      _showToast('Playback failed: $exc');
    }
  }

  Future<void> _stopAudio() async {
    await _audioPlayer.stop();
    if (!mounted) {
      return;
    }
    setState(() {
      _currentPlayingPath = null;
    });
  }

  Future<void> _openSettingsDialog() async {
    final controller = TextEditingController(text: apiUrl);
    await showDialog<void>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Settings'),
          content: TextField(
            controller: controller,
            decoration: const InputDecoration(
              labelText: 'API Base URL',
              hintText: 'https://example.com',
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () async {
                final navigator = Navigator.of(context);
                await _saveApiUrl(controller.text);
                if (mounted) {
                  navigator.pop();
                }
              },
              child: const Text('Save'),
            ),
          ],
        );
      },
    );
    controller.dispose();
  }

  bool _isVideoPath(String path) {
    final lower = path.toLowerCase();
    final idx = lower.lastIndexOf('.');
    if (idx < 0 || idx == lower.length - 1) {
      return false;
    }
    return _videoExtensions.contains(lower.substring(idx + 1));
  }

  Future<void> pickInputFile() async {
    final res = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: _supportedInputExtensions,
    );

    if (res != null && res.files.single.path != null) {
      _setSelectedInput(res.files.single.path!);
      _showToast(_selectedInputIsVideo ? 'Selected video file.' : 'Selected audio file.');
    }
  }

  Future<void> startRecording() async {
    if (_isRecording) {
      return;
    }

    try {
      final hasPermission = await _audioRecorder.hasPermission();
      if (!hasPermission) {
        _showToast('Microphone permission is required.');
        return;
      }

      final tempDir = await getTemporaryDirectory();
      final path =
          '${tempDir.path}/recorded_${DateTime.now().millisecondsSinceEpoch}.wav';

      await _audioRecorder.start(
        const RecordConfig(
          encoder: AudioEncoder.wav,
          sampleRate: 16000,
          numChannels: 1,
          bitRate: 128000,
        ),
        path: path,
      );

      await _amplitudeSub?.cancel();
      _amplitudeSub = _audioRecorder
          .onAmplitudeChanged(const Duration(milliseconds: 140))
          .listen((amplitude) {
        if (!mounted || !_isRecording) {
          return;
        }
        final normalized = ((amplitude.current + 60) / 60).clamp(0.0, 1.0);
        setState(() {
          _recordingLevel = normalized.toDouble();
        });
      });

      _recordingTicker?.cancel();
      _recordingTicker = Timer.periodic(const Duration(seconds: 1), (timer) {
        if (!mounted || !_isRecording) {
          timer.cancel();
          return;
        }
        setState(() {
          _recordingSeconds += 1;
        });
      });

      if (!mounted) {
        return;
      }
      setState(() {
        _isRecording = true;
        _recordingSeconds = 0;
        _recordingLevel = 0.0;
        resultText = 'Recording... tap Stop Recording when done.';
      });
      _showToast('Recording started.');
    } catch (exc) {
      _showToast('Failed to start recording: $exc');
    }
  }

  Future<void> stopRecording() async {
    if (!_isRecording) {
      return;
    }

    try {
      final outputPath = await _audioRecorder.stop();
      await _amplitudeSub?.cancel();
      _amplitudeSub = null;
      _recordingTicker?.cancel();
      _recordingTicker = null;
      if (!mounted) {
        return;
      }

      setState(() {
        _isRecording = false;
        _recordingLevel = 0.0;
      });

      if (outputPath == null || outputPath.isEmpty) {
        _showToast('Recording stopped, but no file was created.');
        return;
      }

      _setSelectedInput(outputPath, fromRecording: true);
      _showToast('Recording saved and selected as input.');
    } catch (exc) {
      if (mounted) {
        setState(() {
          _isRecording = false;
          _recordingLevel = 0.0;
        });
      }
      await _amplitudeSub?.cancel();
      _amplitudeSub = null;
      _recordingTicker?.cancel();
      _recordingTicker = null;
      _showToast('Failed to stop recording: $exc');
    }
  }

  Future<void> analyze() async {
    if (selectedPath == null) {
      return;
    }

    await analyzePath(selectedPath!);
  }

  Future<void> analyzeDenoised() async {
    if (denoisedPath == null) {
      return;
    }

    await analyzePath(denoisedPath!);
  }

  Future<void> analyzePath(String path) async {
    final api = _api;
    if (api == null) {
      _showToast('Loading settings, please wait...');
      return;
    }

    setState(() {
      loading = true;
      lastRequestMs = null;
      resultText = 'Analyzing...';
    });

    try {
      final started = DateTime.now();
      final data = await api.predictFromFilePath(path);
      final elapsed = DateTime.now().difference(started).inMilliseconds;
      final label = data['label'];
      final conf = (data['confidence'] as num).toDouble();
      setState(() {
        predictedLabel = label.toString();
        confidence = conf.clamp(0.0, 1.0);
        lastRequestMs = elapsed;
        resultText = 'Detected: $label\nConfidence: ${(conf * 100).toStringAsFixed(2)}%';
      });
    } catch (exc) {
      setState(() {
        predictedLabel = null;
        confidence = null;
        lastRequestMs = null;
        resultText = 'Failed: $exc';
      });
      _showToast('Analyze request failed.');
    } finally {
      setState(() {
        loading = false;
      });
    }
  }

  Future<void> denoiseAudio() async {
    await denoiseAudioTrained();
  }

  Future<void> denoiseAudioTrained() async {
    final api = _api;
    if (api == null) {
      _showToast('Loading settings, please wait...');
      return;
    }

    if (selectedPath == null) {
      return;
    }

    setState(() {
      loading = true;
      lastRequestMs = null;
      resultText = 'Denoising with trained API...';
    });

    try {
      final started = DateTime.now();
      final output = await api.denoiseToTempFileTrained(selectedPath!);
      final elapsed = DateTime.now().difference(started).inMilliseconds;
      setState(() {
        denoisedPath = output;
        predictedLabel = null;
        confidence = null;
        lastRequestMs = elapsed;
        resultText =
            'Denoised using trained API. File ready. Tap Analyze Denoised.';
      });
      _showToast('Denoised audio (trained API) is ready.');
    } catch (exc) {
      setState(() {
        predictedLabel = null;
        confidence = null;
        lastRequestMs = null;
        resultText = 'Trained denoise failed: $exc';
      });
      _showToast('Trained denoise request failed.');
    } finally {
      setState(() {
        loading = false;
      });
    }
  }

  Future<void> denoiseAudioAuphonic() async {
    final api = _api;
    if (api == null) {
      _showToast('Loading settings, please wait...');
      return;
    }

    if (selectedPath == null) {
      return;
    }

    setState(() {
      loading = true;
      lastRequestMs = null;
      resultText = 'Denoising with Auphonic API...';
    });

    try {
      final started = DateTime.now();
      final output = await api.denoiseToTempFileAuphonic(selectedPath!);
      final elapsed = DateTime.now().difference(started).inMilliseconds;
      setState(() {
        denoisedPath = output;
        predictedLabel = null;
        confidence = null;
        lastRequestMs = elapsed;
        resultText = 'Denoised using Auphonic API. File ready. Tap Analyze Denoised.';
      });
      _showToast('Denoised audio (Auphonic API) is ready.');
    } catch (exc) {
      setState(() {
        predictedLabel = null;
        confidence = null;
        lastRequestMs = null;
        resultText = 'Auphonic denoise failed: $exc';
      });
      _showToast('Auphonic denoise request failed.');
    } finally {
      setState(() {
        loading = false;
      });
    }
  }

  Future<void> runSmartPipeline() async {
    final api = _api;
    if (api == null) {
      _showToast('Loading settings, please wait...');
      return;
    }

    if (selectedPath == null) {
      return;
    }

    setState(() {
      loading = true;
      predictedLabel = null;
      confidence = null;
      lastRequestMs = null;
      resultText = 'Denoising and analyzing...';
    });

    try {
      final started = DateTime.now();
      final output = await api.denoiseToTempFile(selectedPath!);
      final data = await api.predictFromFilePath(output);
      final elapsed = DateTime.now().difference(started).inMilliseconds;
      final label = data['label'];
      final conf = (data['confidence'] as num).toDouble();

      setState(() {
        denoisedPath = output;
        predictedLabel = label.toString();
        confidence = conf.clamp(0.0, 1.0);
        lastRequestMs = elapsed;
        resultText = 'Smart result: $label\nConfidence: ${(conf * 100).toStringAsFixed(2)}%';
      });
      _showToast('Smart pipeline completed.');
    } catch (exc) {
      setState(() {
        predictedLabel = null;
        confidence = null;
        lastRequestMs = null;
        resultText = 'Smart analysis failed: $exc';
      });
      _showToast('Smart pipeline failed.');
    } finally {
      setState(() {
        loading = false;
      });
    }
  }

  Widget _buildPathCard(String title, String value) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFF6F8F7),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFDFE7E3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 6),
          Text(value, style: const TextStyle(fontSize: 13)),
        ],
      ),
    );
  }

  String _filenameFromPath(String path) {
    final normalized = path.replaceAll('\\\\', '/');
    final segments = normalized.split('/');
    return segments.isEmpty ? path : segments.last;
  }

  Widget _buildPredictionCard() {
    final conf = confidence;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFEEF6F2),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFB6DCCB)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            predictedLabel == null ? 'No prediction yet' : 'Prediction: $predictedLabel',
            style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 10),
          LinearProgressIndicator(
            value: conf,
            minHeight: 8,
            borderRadius: BorderRadius.circular(8),
            color: const Color(0xFF0C8A5A),
            backgroundColor: const Color(0xFFCEE9DC),
          ),
          const SizedBox(height: 6),
          Text(
            conf == null ? 'Confidence: -' : 'Confidence: ${(conf * 100).toStringAsFixed(2)}%',
            style: const TextStyle(fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 4),
          Text(
            lastRequestMs == null ? 'Request Time: -' : 'Request Time: $lastRequestMs ms',
            style: const TextStyle(fontWeight: FontWeight.w500),
          ),
        ],
      ),
    );
  }

  Widget _buildRecordingStatusCard() {
    final bars = List<Widget>.generate(18, (index) {
      final threshold = (index + 1) / 18;
      final isActive = _isRecording && _recordingLevel >= threshold;
      final height = 6.0 + (index % 6) * 2.0;
      return AnimatedContainer(
        duration: const Duration(milliseconds: 120),
        width: 6,
        height: height,
        decoration: BoxDecoration(
          color: isActive ? const Color(0xFF0C8A5A) : const Color(0xFFCFE6DB),
          borderRadius: BorderRadius.circular(4),
        ),
      );
    });

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFF2F8F5),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFC7E2D4)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                _isRecording ? Icons.fiber_manual_record : Icons.radio_button_unchecked,
                color: _isRecording ? Colors.red : Colors.grey,
                size: 16,
              ),
              const SizedBox(width: 8),
              Text(
                _isRecording ? 'Recording in progress' : 'Recorder idle',
                style: const TextStyle(fontWeight: FontWeight.w700),
              ),
              const Spacer(),
              Text(
                _formatRecordingTime(_recordingSeconds),
                style: const TextStyle(fontWeight: FontWeight.w700),
              ),
            ],
          ),
          const SizedBox(height: 10),
          LinearProgressIndicator(
            value: _isRecording ? _recordingLevel : 0,
            minHeight: 7,
            borderRadius: BorderRadius.circular(8),
            color: const Color(0xFF0C8A5A),
            backgroundColor: const Color(0xFFD9ECE3),
          ),
          const SizedBox(height: 10),
          SizedBox(
            height: 18,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: bars,
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Record, Denoise, Analyze'),
        centerTitle: true,
        actions: [
          IconButton(
            onPressed: loading ? null : _openSettingsDialog,
            icon: const Icon(Icons.settings),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                'Choose input source',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w700,
                    ),
              ),
              const SizedBox(height: 10),
              Row(
                children: [
                  Expanded(
                    child: FilledButton.tonalIcon(
                      onPressed: loading || _isRecording ? null : pickInputFile,
                      icon: const Icon(Icons.upload_file),
                      label: const Text('Pick Audio or Video'),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 10),
              Row(
                children: [
                  Expanded(
                    child: FilledButton.icon(
                      onPressed: loading || _isRecording ? null : startRecording,
                      icon: const Icon(Icons.mic),
                      label: const Text('Start Recording'),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: FilledButton.tonalIcon(
                      onPressed: _isRecording ? stopRecording : null,
                      icon: const Icon(Icons.stop_circle_outlined),
                      label: const Text('Stop Recording'),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              _buildRecordingStatusCard(),
              const SizedBox(height: 10),
              _buildPathCard('Selected Input', selectedPath ?? 'No input selected'),
              const SizedBox(height: 10),
              _buildPathCard('Last Recording', recordedPath ?? 'No recording yet'),
              const SizedBox(height: 10),
              _buildPathCard('Denoised File', denoisedPath ?? 'No denoised file yet'),
              const SizedBox(height: 10),
              Text(
                'API: $apiUrl',
                style: const TextStyle(fontSize: 12, color: Colors.black54),
              ),
              const SizedBox(height: 16),
              FilledButton.icon(
                onPressed: loading || selectedPath == null ? null : runSmartPipeline,
                icon: const Icon(Icons.auto_awesome),
                label: const Text('Denoise + Analyze Selected Input'),
              ),
              const SizedBox(height: 10),
              FilledButton.tonalIcon(
                onPressed: loading || selectedPath == null || _selectedInputIsVideo
                    ? null
                    : () => _playAudio(selectedPath!),
                icon: const Icon(Icons.play_arrow),
                label: Text(
                  selectedPath == null
                      ? 'Play Selected Input'
                      : 'Play Selected Input (${_filenameFromPath(selectedPath!)})',
                ),
              ),
              const SizedBox(height: 10),
              FilledButton.tonalIcon(
                onPressed: loading || denoisedPath == null ? null : () => _playAudio(denoisedPath!),
                icon: const Icon(Icons.play_circle),
                label: const Text('Play Denoised'),
              ),
              const SizedBox(height: 10),
              FilledButton.tonalIcon(
                onPressed: _currentPlayingPath == null ? null : _stopAudio,
                icon: const Icon(Icons.stop),
                label: const Text('Stop Playback'),
              ),
              const SizedBox(height: 10),
              FilledButton.tonal(
                onPressed: loading || selectedPath == null ? null : denoiseAudioTrained,
                child: const Text('Denoise (My Trained API)'),
              ),
              const SizedBox(height: 10),
              FilledButton.tonal(
                onPressed: loading || selectedPath == null ? null : denoiseAudioAuphonic,
                child: const Text('Denoise (Auphonic API)'),
              ),
              const SizedBox(height: 10),
              FilledButton(
                onPressed: loading || selectedPath == null ? null : analyze,
                child: const Text('Analyze Original'),
              ),
              const SizedBox(height: 10),
              FilledButton(
                onPressed: loading || denoisedPath == null ? null : analyzeDenoised,
                child: const Text('Analyze Denoised'),
              ),
              const SizedBox(height: 18),
              if (loading)
                const Center(
                  child: CircularProgressIndicator(),
                ),
              const SizedBox(height: 12),
              _buildPredictionCard(),
              const SizedBox(height: 12),
              Text(
                resultText,
                style: const TextStyle(fontSize: 15),
              ),
              const SizedBox(height: 10),
              const Text(
                'Made by Ravi Kant & Raushan',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
