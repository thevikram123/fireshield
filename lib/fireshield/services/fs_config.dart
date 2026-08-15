/// Runtime configuration for the FireShield AI services.
///
/// The Cloudflare Worker URL is injected at build time so it never has to be
/// hard-coded, and the Groq key never lives in the app at all:
///
///   flutter run   -t lib/fireshield_main.dart -d chrome \
///     --dart-define=FIRESHIELD_WORKER_URL=https://fireshield-groq-gateway.<sub>.workers.dev
///   flutter build web -t lib/fireshield_main.dart --release \
///     --dart-define=FIRESHIELD_WORKER_URL=https://…workers.dev --base-href /<repo>/
library;

/// Base URL of the deployed Worker gateway. Empty when not configured — the AI
/// features then surface an explicit "not configured" state instead of faking
/// results.
const String workerBaseUrl =
    String.fromEnvironment('FIRESHIELD_WORKER_URL', defaultValue: '');

bool get isWorkerConfigured => workerBaseUrl.isNotEmpty;
