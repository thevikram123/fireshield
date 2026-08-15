library;

import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import 'fs_config.dart';
import 'fs_groq_service.dart';

class FsPlanResult {
  final Map<String, dynamic> topology;
  final Map<String, dynamic> commercialModel;
  final Map<String, dynamic> guidance;
  final Map<String, dynamic> visionAdvisory;
  final Map<String, dynamic> metrics;
  final Map<String, dynamic> artifacts;
  final Map<String, dynamic> compliance;
  final List<String> warnings;

  const FsPlanResult({
    required this.topology,
    required this.commercialModel,
    required this.guidance,
    required this.visionAdvisory,
    required this.metrics,
    required this.artifacts,
    required this.compliance,
    required this.warnings,
  });

  factory FsPlanResult.fromJson(Map<String, dynamic> json) => FsPlanResult(
        topology: Map<String, dynamic>.from(json['topology'] as Map? ?? {}),
        commercialModel:
            Map<String, dynamic>.from(json['commercialModel'] as Map? ?? {}),
        guidance: Map<String, dynamic>.from(json['guidance'] as Map? ?? {}),
        visionAdvisory:
            Map<String, dynamic>.from(json['visionAdvisory'] as Map? ?? {}),
        metrics: Map<String, dynamic>.from(json['metrics'] as Map? ?? {}),
        artifacts: Map<String, dynamic>.from(json['artifacts'] as Map? ?? {}),
        compliance: Map<String, dynamic>.from(json['compliance'] as Map? ?? {}),
        warnings: (json['warnings'] as List? ?? const [])
            .map((item) => item.toString())
            .toList(),
      );
}

class FsPlanService {
  FsPlanService({http.Client? client, String? baseUrl})
      : _client = client ?? http.Client(),
        _base = (baseUrl ?? workerBaseUrl).replaceAll(RegExp(r'/+$'), '');

  final http.Client _client;
  final String _base;

  bool get isConfigured => _base.isNotEmpty;

  Future<FsPlanResult> convert({
    required Uint8List bytes,
    required String filename,
    required String mimeType,
    int page = 0,
    String? overall,
    Map<String, dynamic> buildingProfile = const {},
  }) async {
    if (!isConfigured) {
      throw const FsServiceException(
          'Plan service not configured — set FIRESHIELD_WORKER_URL.');
    }
    if (bytes.isEmpty) {
      throw const FsServiceException('The selected file is empty.');
    }
    if (bytes.length > 25 * 1024 * 1024) {
      throw const FsServiceException('The selected plan exceeds 25 MB.');
    }
    final request =
        http.MultipartRequest('POST', Uri.parse('$_base/plan/convert'))
          ..fields['page'] = '$page'
          ..fields['require_qwen'] = 'true'
          ..fields['building_profile'] = jsonEncode(buildingProfile);
    if (isSupabaseConfigured) {
      final token = Supabase.instance.client.auth.currentSession?.accessToken;
      if (token != null && token.isNotEmpty) {
        request.headers['Authorization'] = 'Bearer $token';
      }
    }
    if (overall != null && overall.trim().isNotEmpty) {
      request.fields['overall'] = overall.trim();
    }
    request.files.add(http.MultipartFile.fromBytes(
      'file',
      bytes,
      filename: filename,
      contentType: MediaType.parse(mimeType),
    ));

    try {
      final streamed =
          await _client.send(request).timeout(const Duration(minutes: 3));
      final response = await http.Response.fromStream(streamed);
      final decoded = jsonDecode(utf8.decode(response.bodyBytes));
      if (decoded is! Map<String, dynamic>) {
        throw const FsServiceException(
            'Plan service returned an invalid response.');
      }
      if (response.statusCode >= 400 || decoded['error'] != null) {
        throw FsServiceException(
          decoded['detail']?.toString() ??
              decoded['error']?.toString() ??
              'Plan conversion failed.',
          status: response.statusCode,
        );
      }
      return FsPlanResult.fromJson(decoded);
    } on TimeoutException {
      throw const FsServiceException(
          'Plan processing timed out. The conversion service may be starting; retry once.');
    } on FsServiceException {
      rethrow;
    } catch (error) {
      throw FsServiceException('Could not process the plan: $error');
    }
  }

  void dispose() => _client.close();
}
