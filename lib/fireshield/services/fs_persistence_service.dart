/// Supabase persistence for FireShield evidence, model runs and findings.
///
/// No method reports success until the remote operation succeeds. The service
/// deliberately has no in-memory fallback because that would make saved
/// evidence appear durable when it is not.
library;

import 'dart:typed_data';

import 'package:supabase_flutter/supabase_flutter.dart';

import '../data/fs_models.dart';
import 'fs_config.dart';

class FsPersistenceException implements Exception {
  final String message;
  const FsPersistenceException(this.message);
  @override
  String toString() => 'FsPersistenceException: $message';
}

class FsPersistenceService {
  FsPersistenceService({SupabaseClient? client}) : _client = client;

  final SupabaseClient? _client;

  SupabaseClient get client {
    if (!isSupabaseConfigured && _client == null) {
      throw const FsPersistenceException(
          'Database is not configured for this build.');
    }
    return _client ?? Supabase.instance.client;
  }

  User get currentUser =>
      client.auth.currentUser ??
      (throw const FsPersistenceException(
          'Sign in before saving an assessment.'));

  bool get isSignedIn =>
      isSupabaseConfigured && client.auth.currentSession != null;

  Future<String> currentOrganisationId() async {
    final user = currentUser;
    try {
      final row = await client
          .from('organisation_members')
          .select('organisation_id')
          .eq('user_id', user.id)
          .limit(1)
          .maybeSingle();
      if (row == null) {
        throw const FsPersistenceException(
            'Your account is not assigned to a FireShield organisation.');
      }
      return row['organisation_id'].toString();
    } on PostgrestException catch (error) {
      throw FsPersistenceException(error.message);
    }
  }

  Future<void> signInWithPassword(String email, String password) async {
    try {
      await client.auth.signInWithPassword(email: email, password: password);
    } on AuthException catch (error) {
      throw FsPersistenceException(error.message);
    } on PostgrestException catch (error) {
      throw FsPersistenceException(error.message);
    }
  }

  Future<bool> signUpOrganisationAdmin({
    required String email,
    required String password,
    required String displayName,
    required String organisationName,
  }) async {
    try {
      final response = await client.auth.signUp(
        email: email,
        password: password,
        data: {
          'display_name': displayName.trim(),
          // Used only to provision the user's first organisation after Auth
          // verifies the session. It is never trusted for RLS authorization.
          'pending_organisation_name': organisationName.trim(),
        },
      );
      if (response.session != null) {
        await _provisionFirstOrganisationIfNeeded();
        return true;
      }
      return false;
    } on AuthException catch (error) {
      throw FsPersistenceException(error.message);
    } on PostgrestException catch (error) {
      throw FsPersistenceException(error.message);
    }
  }

  Future<void> signOut() async => client.auth.signOut();

  Future<FsUser> currentAppUser() async {
    final user = currentUser;
    try {
      var membership = await client
          .from('organisation_members')
          .select(
              'organisation_id, role, organisations(name), profiles(display_name)')
          .eq('user_id', user.id)
          .limit(1)
          .maybeSingle();
      if (membership == null) {
        await _provisionFirstOrganisationIfNeeded();
        membership = await client
            .from('organisation_members')
            .select(
                'organisation_id, role, organisations(name), profiles(display_name)')
            .eq('user_id', user.id)
            .limit(1)
            .maybeSingle();
        if (membership == null) {
          throw const FsPersistenceException(
              'Your account is not assigned to a FireShield organisation.');
        }
      }
      final org =
          membership['organisations'] as Map<String, dynamic>? ?? const {};
      final profile =
          membership['profiles'] as Map<String, dynamic>? ?? const {};
      final name =
          (profile['display_name']?.toString().trim().isNotEmpty ?? false)
              ? profile['display_name'].toString().trim()
              : (user.email ?? 'FireShield user');
      final words = name.split(RegExp(r'\s+')).where((word) => word.isNotEmpty);
      final initials =
          words.take(2).map((word) => word[0].toUpperCase()).join();
      return FsUser(
        id: user.id.hashCode,
        name: name,
        role: FsRoleInfo.fromKey(membership['role']?.toString() ?? 'auditor'),
        email: user.email ?? '',
        facility: '',
        initials: initials.isEmpty ? 'FS' : initials,
        dept: '',
        org: org['name']?.toString() ?? '',
      );
    } on PostgrestException catch (error) {
      throw FsPersistenceException(error.message);
    }
  }

  Future<void> _provisionFirstOrganisationIfNeeded() async {
    final user = currentUser;
    final existing = await client
        .from('organisation_members')
        .select('organisation_id')
        .eq('user_id', user.id)
        .limit(1)
        .maybeSingle();
    if (existing != null) return;
    final metadata = user.userMetadata ?? const <String, dynamic>{};
    final organisationName =
        metadata['pending_organisation_name']?.toString().trim() ?? '';
    if (organisationName.length < 2) return;
    await createOrganisation(organisationName);
    final displayName = metadata['display_name']?.toString().trim() ?? '';
    if (displayName.isNotEmpty) {
      await client
          .from('profiles')
          .update({'display_name': displayName}).eq('id', user.id);
    }
  }

  Future<String> createOrganisation(String name) async {
    currentUser;
    try {
      final value = await client.rpc('create_organisation', params: {
        'org_name': name,
      });
      return value.toString();
    } on PostgrestException catch (error) {
      throw FsPersistenceException(error.message);
    }
  }

  Future<String> createAssessment({
    required String organisationId,
    required String kind,
    required String title,
    String? facilityId,
    Map<String, dynamic> buildingProfile = const {},
  }) async {
    final user = currentUser;
    try {
      final row = await client
          .from('assessments')
          .insert({
            'organisation_id': organisationId,
            'facility_id': facilityId,
            'created_by': user.id,
            'kind': kind,
            'status': 'draft',
            'title': title,
            'building_profile': buildingProfile,
          })
          .select('id')
          .single();
      return row['id'].toString();
    } on PostgrestException catch (error) {
      throw FsPersistenceException(error.message);
    }
  }

  Future<String> uploadArtifact({
    required String organisationId,
    required String assessmentId,
    required String filename,
    required String kind,
    required String mimeType,
    required Uint8List bytes,
    String? sourceArtifactId,
  }) async {
    currentUser;
    final safeName = filename.replaceAll(RegExp(r'[^A-Za-z0-9._-]'), '_');
    final path = '$organisationId/$assessmentId/$safeName';
    try {
      await client.storage.from('assessment-artifacts').uploadBinary(
            path,
            bytes,
            fileOptions: FileOptions(contentType: mimeType, upsert: false),
          );
      final row = await client
          .from('artifacts')
          .insert({
            'assessment_id': assessmentId,
            'kind': kind,
            'storage_path': path,
            'mime_type': mimeType,
            'byte_size': bytes.length,
            'source_artifact_id': sourceArtifactId,
          })
          .select('id')
          .single();
      return row['id'].toString();
    } on StorageException catch (error) {
      throw FsPersistenceException(error.message);
    } on PostgrestException catch (error) {
      // The file exists but metadata failed. Surface the failure so the UI does
      // not claim the assessment is safely recorded.
      throw FsPersistenceException(
          'Artifact uploaded but metadata was not saved: ${error.message}');
    }
  }

  Future<String> beginModelRun({
    required String assessmentId,
    required String provider,
    required String model,
    required String purpose,
    List<String> inputArtifactIds = const [],
  }) async {
    currentUser;
    try {
      final row = await client
          .from('model_runs')
          .insert({
            'assessment_id': assessmentId,
            'provider': provider,
            'model': model,
            'purpose': purpose,
            'status': 'running',
            'input_artifact_ids': inputArtifactIds,
          })
          .select('id')
          .single();
      return row['id'].toString();
    } on PostgrestException catch (error) {
      throw FsPersistenceException(error.message);
    }
  }

  Future<void> saveSiteResult({
    required String assessmentId,
    required String modelRunId,
    required FsAuditRun result,
    required int latencyMs,
  }) async {
    currentUser;
    try {
      if (result.detected.isNotEmpty) {
        await client.from('detections').insert(result.detected
            .map((item) => {
                  'assessment_id': assessmentId,
                  'model_run_id': modelRunId,
                  ...item.toJson(),
                  'reviewed': false,
                })
            .toList());
      }
      for (final finding in result.findings) {
        final row = await client
            .from('findings')
            .insert({
              'assessment_id': assessmentId,
              'model_run_id': modelRunId,
              ...finding.toJson()
                ..remove('clauseId')
                ..remove('page'),
            })
            .select('id')
            .single();
        if (finding.clauseId.isNotEmpty) {
          await client.from('finding_citations').insert({
            'finding_id': row['id'],
            'clause_id': finding.clauseId,
            'page': finding.page ?? 0,
          });
        }
      }
      await client.from('model_runs').update({
        'status': 'succeeded',
        'latency_ms': latencyMs,
        'output': {
          'score': result.score,
          'summary': result.occupancySummary,
        },
      }).eq('id', modelRunId);
      await client.from('assessments').update({
        'status': 'complete',
        'score': result.score,
        'summary': result.occupancySummary,
        'completed_at': DateTime.now().toUtc().toIso8601String(),
        'updated_at': DateTime.now().toUtc().toIso8601String(),
      }).eq('id', assessmentId);
    } on PostgrestException catch (error) {
      throw FsPersistenceException(error.message);
    }
  }

  Future<void> savePlanResult({
    required String assessmentId,
    required String modelRunId,
    required Map<String, dynamic> compliance,
    required Map<String, dynamic> metrics,
    required Map<String, dynamic> guidance,
    required int latencyMs,
  }) async {
    currentUser;
    final findings = compliance['findings'] as List? ?? const [];
    try {
      for (final item in findings) {
        if (item is! Map) continue;
        final finding = Map<String, dynamic>.from(item);
        final row = await client
            .from('findings')
            .insert({
              'assessment_id': assessmentId,
              'model_run_id': modelRunId,
              'system': finding['check']?.toString() ?? 'Plan check',
              'status': finding['status']?.toString() ?? 'cannot_verify',
              'severity': finding['severity']?.toString() ?? 'major',
              'observed': finding['observed']?.toString(),
              'required': finding['required']?.toString(),
              'rationale': finding['rationale']?.toString(),
            })
            .select('id')
            .single();
        final clauseId = finding['clauseId']?.toString() ?? '';
        final page = finding['page'];
        if (clauseId.isNotEmpty && page is num) {
          await client.from('finding_citations').insert({
            'finding_id': row['id'],
            'clause_id': clauseId,
            'page': page.toInt(),
          });
        }
      }
      await client.from('model_runs').update({
        'status': 'succeeded',
        'latency_ms': latencyMs,
        'output': {
          'compliance': compliance,
          'metrics': metrics,
          'qwenGuidance': guidance,
        },
      }).eq('id', modelRunId);
      final score = compliance['score'];
      await client.from('assessments').update({
        'status': 'complete',
        if (score is num) 'score': score,
        'summary': compliance['planSummary']?.toString() ?? '',
        'completed_at': DateTime.now().toUtc().toIso8601String(),
        'updated_at': DateTime.now().toUtc().toIso8601String(),
      }).eq('id', assessmentId);
    } on PostgrestException catch (error) {
      throw FsPersistenceException(error.message);
    }
  }

  Future<List<Map<String, dynamic>>> assessmentHistory() async {
    currentUser;
    try {
      return await client
          .from('assessments')
          .select(
              'id, kind, title, status, score, summary, created_at, completed_at')
          .order('created_at', ascending: false);
    } on PostgrestException catch (error) {
      throw FsPersistenceException(error.message);
    }
  }

  Future<String> signedArtifactUrl(String path,
          {Duration validFor = const Duration(minutes: 10)}) async =>
      client.storage
          .from('assessment-artifacts')
          .createSignedUrl(path, validFor.inSeconds);
}
