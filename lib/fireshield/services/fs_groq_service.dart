/// Client for the FireShield secure Groq gateway (Cloudflare Worker).
///
/// The app never holds GROQ_API_KEY — it POSTs to the Worker, which adds the
/// key server-side and (for reasoning) queries the NBCS 2026 graph live.
/// On any failure this throws [FsServiceException]; it never substitutes fake
/// compliance content (skill rule #8).
library;

import 'dart:convert';

import 'package:http/http.dart' as http;

import '../data/fs_models.dart';
import 'fs_config.dart';

class FsServiceException implements Exception {
  final String message;
  final int? status;
  const FsServiceException(this.message, {this.status});
  @override
  String toString() => 'FsServiceException($status): $message';
}

class FsGroqService {
  FsGroqService({http.Client? client, String? baseUrl})
      : _client = client ?? http.Client(),
        _base = (baseUrl ?? workerBaseUrl).replaceAll(RegExp(r'/+$'), '');

  final http.Client _client;
  final String _base;

  static const _timeout = Duration(seconds: 60);

  bool get isConfigured => _base.isNotEmpty;

  Uri _uri(String path) => Uri.parse('$_base$path');

  Future<Map<String, dynamic>> _post(String path, Map<String, dynamic> body) async {
    if (!isConfigured) {
      throw const FsServiceException(
          'AI service not configured — set FIRESHIELD_WORKER_URL at build time.');
    }
    http.Response res;
    try {
      res = await _client
          .post(_uri(path),
              headers: const {'Content-Type': 'application/json'},
              body: jsonEncode(body))
          .timeout(_timeout);
    } catch (e) {
      throw FsServiceException('Could not reach AI service: $e');
    }
    Map<String, dynamic>? decoded;
    try {
      decoded = jsonDecode(res.body) as Map<String, dynamic>;
    } catch (_) {
      decoded = null;
    }
    if (res.statusCode >= 400 || decoded == null) {
      final msg = decoded?['error']?.toString() ?? 'request failed';
      throw FsServiceException(msg, status: res.statusCode);
    }
    if (decoded['error'] != null) {
      throw FsServiceException(decoded['error'].toString(), status: res.statusCode);
    }
    return decoded;
  }

  /// Vision detection: base64 data URLs → observed equipment (server-side Qwen).
  /// Works on every platform incl. mobile browsers (unlike in-browser CLIPSeg).
  Future<List<DetectedEquipment>> visionDetect(List<String> imageDataUrls,
      {String? prompt}) async {
    if (imageDataUrls.isEmpty) return const [];
    final body = <String, dynamic>{'images': imageDataUrls.take(5).toList()};
    if (prompt != null && prompt.isNotEmpty) body['prompt'] = prompt;
    final res = await _post('/groq/vision', body);
    final list = res['detections'] as List? ?? const [];
    return list
        .whereType<Map<String, dynamic>>()
        .map((m) => DetectedEquipment.fromJson({...m, 'source': m['source'] ?? 'qwen'}))
        .toList();
  }

  /// Direct NBC graph query (Regulations phase). Returns cited clauses.
  Future<List<NbcClause>> nbcQuery(String question,
      {String seedTerms = '', int k = 8, int hops = 1}) async {
    final res = await _post('/nbc/query', {
      'question': question,
      if (seedTerms.isNotEmpty) 'seed_terms': seedTerms,
      'k': k,
      'hops': hops,
    });
    final results = res['results'] as List? ?? const [];
    return results.whereType<Map<String, dynamic>>().map(NbcClause.fromJson).toList();
  }

  /// Compliance reasoning: gpt-oss decides mandatory systems, queries the graph
  /// live for each, compares vs observed, and returns findings + score.
  Future<FsAuditRun> reasonCompliance({
    required String occupancyGroup,
    required Map<String, dynamic> buildingProfile,
    required List<DetectedEquipment> detected,
    List<String> docs = const [],
  }) async {
    final res = await _post('/groq/reason', {
      'buildingProfile': buildingProfile,
      'detected': detected.map((d) => d.toJson()).toList(),
      'docs': docs,
    });
    return FsAuditRun.fromVerdict(
      res,
      occupancyGroup: occupancyGroup,
      buildingProfile: buildingProfile,
      detected: detected,
      docs: docs,
    );
  }

  void dispose() => _client.close();
}
