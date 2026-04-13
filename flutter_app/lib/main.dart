import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

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
  final api = ApiService('https://your-api-url.onrender.com');

  String? selectedPath;
  String? denoisedPath;
  String resultText = 'Pick an audio file and run denoise + analysis.';
  String? predictedLabel;
  double? confidence;
  bool loading = false;

  Future<void> pickAudio() async {
    final res = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['wav', 'mp3', 'm4a', 'flac', 'ogg'],
    );

    if (res != null && res.files.single.path != null) {
      setState(() {
        selectedPath = res.files.single.path!;
      });
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
    setState(() {
      loading = true;
      resultText = 'Analyzing...';
    });

    try {
      final data = await api.predictFromFilePath(path);
      final label = data['label'];
      final conf = (data['confidence'] as num).toDouble();
      setState(() {
        predictedLabel = label.toString();
        confidence = conf.clamp(0.0, 1.0);
        resultText = 'Detected: $label\nConfidence: ${(conf * 100).toStringAsFixed(2)}%';
      });
    } catch (exc) {
      setState(() {
        predictedLabel = null;
        confidence = null;
        resultText = 'Failed: $exc';
      });
    } finally {
      setState(() {
        loading = false;
      });
    }
  }

  Future<void> denoiseAudio() async {
    if (selectedPath == null) {
      return;
    }

    setState(() {
      loading = true;
      resultText = 'Denoising...';
    });

    try {
      final output = await api.denoiseToTempFile(selectedPath!);
      setState(() {
        denoisedPath = output;
        predictedLabel = null;
        confidence = null;
        resultText = 'Denoised file ready. Tap Analyze Denoised.';
      });
    } catch (exc) {
      setState(() {
        predictedLabel = null;
        confidence = null;
        resultText = 'Denoise failed: $exc';
      });
    } finally {
      setState(() {
        loading = false;
      });
    }
  }

  Future<void> runSmartPipeline() async {
    if (selectedPath == null) {
      return;
    }

    setState(() {
      loading = true;
      predictedLabel = null;
      confidence = null;
      resultText = 'Denoising and analyzing...';
    });

    try {
      final output = await api.denoiseToTempFile(selectedPath!);
      final data = await api.predictFromFilePath(output);
      final label = data['label'];
      final conf = (data['confidence'] as num).toDouble();

      setState(() {
        denoisedPath = output;
        predictedLabel = label.toString();
        confidence = conf.clamp(0.0, 1.0);
        resultText = 'Smart result: $label\nConfidence: ${(conf * 100).toStringAsFixed(2)}%';
      });
    } catch (exc) {
      setState(() {
        predictedLabel = null;
        confidence = null;
        resultText = 'Smart analysis failed: $exc';
      });
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
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Audio Denoise and Analyzer'),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              ElevatedButton.icon(
                onPressed: loading ? null : pickAudio,
                icon: const Icon(Icons.audio_file),
                label: const Text('Pick Audio File'),
              ),
              const SizedBox(height: 12),
              _buildPathCard('Selected File', selectedPath ?? 'No file selected'),
              const SizedBox(height: 10),
              _buildPathCard('Denoised File', denoisedPath ?? 'No denoised file yet'),
              const SizedBox(height: 16),
              FilledButton.icon(
                onPressed: loading || selectedPath == null ? null : runSmartPipeline,
                icon: const Icon(Icons.auto_awesome),
                label: const Text('Smart Analyze (Denoise + Predict)'),
              ),
              const SizedBox(height: 10),
              FilledButton.tonal(
                onPressed: loading || selectedPath == null ? null : denoiseAudio,
                child: const Text('Denoise Only'),
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
            ],
          ),
        ),
      ),
    );
  }
}
