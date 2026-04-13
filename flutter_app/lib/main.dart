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
  String resultText = 'Pick an audio file and analyze.';
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
        resultText = 'Detected: $label\nConfidence: ${(conf * 100).toStringAsFixed(2)}%';
      });
    } catch (exc) {
      setState(() {
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
        resultText = 'Denoised file ready. Tap Analyze Denoised.';
      });
    } catch (exc) {
      setState(() {
        resultText = 'Denoise failed: $exc';
      });
    } finally {
      setState(() {
        loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Audio Analyzer')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            ElevatedButton(
              onPressed: loading ? null : pickAudio,
              child: const Text('Pick Audio File'),
            ),
            const SizedBox(height: 12),
            Text(selectedPath ?? 'No file selected'),
            const SizedBox(height: 8),
            Text(denoisedPath == null ? 'No denoised file yet' : 'Denoised: $denoisedPath'),
            const SizedBox(height: 20),
            FilledButton.tonal(
              onPressed: loading || selectedPath == null ? null : denoiseAudio,
              child: const Text('Denoise Audio'),
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
            const SizedBox(height: 24),
            if (loading) const CircularProgressIndicator(),
            const SizedBox(height: 16),
            Text(
              resultText,
              style: const TextStyle(fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }
}
