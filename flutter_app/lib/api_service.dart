import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

class ApiService {
  final String baseUrl;
  ApiService(this.baseUrl);

  Future<Map<String, dynamic>> predictFromFilePath(String filePath) async {
    final uri = Uri.parse('$baseUrl/predict');
    final request = http.MultipartRequest('POST', uri);
    request.files.add(await http.MultipartFile.fromPath('file', filePath));

    final streamed = await request.send();
    final response = await http.Response.fromStream(streamed);

    if (response.statusCode == 200) {
      return jsonDecode(response.body) as Map<String, dynamic>;
    }

    throw Exception('API error ${response.statusCode}: ${response.body}');
  }

  Future<String> denoiseToTempFile(String filePath) async {
    final uri = Uri.parse('$baseUrl/denoise');
    final request = http.MultipartRequest('POST', uri);
    request.files.add(await http.MultipartFile.fromPath('file', filePath));

    final streamed = await request.send();
    final response = await http.Response.fromStream(streamed);

    if (response.statusCode != 200) {
      throw Exception('API error ${response.statusCode}: ${response.body}');
    }

    final tempDir = await getTemporaryDirectory();
    final output = File('${tempDir.path}/denoised_${DateTime.now().millisecondsSinceEpoch}.wav');
    await output.writeAsBytes(response.bodyBytes, flush: true);
    return output.path;
  }
}
