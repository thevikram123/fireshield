/// Client for the FireShield secure Groq gateway (Cloudflare Worker).
///
/// The app never holds GROQ_API_KEY — it POSTs to the Worker, which adds the
/// key server-side and (for reasoning) queries the NBCS 2026 graph live.
/// On any failure this throws [FsServiceException]; it never substitutes fake
/// compliance content (skill rule #8).
library;

import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:supabase_flutter/supabase_flutter.dart';

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

  Future<Map<String, dynamic>> _post(
      String path, Map<String, dynamic> body) async {
    if (!isConfigured) {
      throw const FsServiceException(
          'AI service not configured — set FIRESHIELD_WORKER_URL at build time.');
    }
    http.Response res;
    try {
      final headers = <String, String>{'Content-Type': 'application/json'};
      if (isSupabaseConfigured) {
        final token = Supabase.instance.client.auth.currentSession?.accessToken;
        if (token != null && token.isNotEmpty) {
          headers['Authorization'] = 'Bearer $token';
        }
      }
      res = await _client
          .post(_uri(path), headers: headers, body: jsonEncode(body))
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
      throw FsServiceException(decoded['error'].toString(),
          status: res.statusCode);
    }
    return decoded;
  }

  /// Grounded chat through the Worker. The Worker adds relevant NBCS 2026
  /// graph results before calling gpt-oss; the browser never receives the key.
  Future<String> chat(List<Map<String, String>> messages) async {
    final res = await _post('/groq/chat', {'messages': messages});
    return res['content']?.toString() ?? '';
  }

  /// Vision detection: base64 data URLs → observed equipment (server-side Qwen).
  /// Works on every platform incl. mobile browsers (unlike in-browser CLIPSeg).
  Future<List<DetectedEquipment>> visionDetect(List<String> imageDataUrls,
      {String? prompt, Map<String, dynamic>? evidenceContext}) async {
    if (imageDataUrls.isEmpty) return const [];
    final body = <String, dynamic>{'images': imageDataUrls.take(5).toList()};
    if (prompt != null && prompt.isNotEmpty) body['prompt'] = prompt;
    if (evidenceContext != null) body['evidenceContext'] = evidenceContext;
    final res = await _post('/groq/vision', body);
    final list = res['detections'] as List? ?? const [];
    return list
        .whereType<Map<String, dynamic>>()
        .map((m) =>
            DetectedEquipment.fromJson({...m, 'source': m['source'] ?? 'qwen'}))
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
    return results
        .whereType<Map<String, dynamic>>()
        .map(NbcClause.fromJson)
        .toList();
  }

  /// Compliance reasoning: gpt-oss decides mandatory systems, queries the graph
  /// live for each, compares vs observed, and returns findings + score.
  Future<FsAuditRun> reasonCompliance({
    required String occupancyGroup,
    required Map<String, dynamic> buildingProfile,
    required List<DetectedEquipment> detected,
    List<String> docs = const [],
    Map<String, dynamic> evidenceContext = const {},
  }) async {
    final res = await _post('/groq/reason', {
      'buildingProfile': buildingProfile,
      'detected': detected.map((d) => d.toJson()).toList(),
      'docs': docs,
      'evidenceContext': evidenceContext,
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
