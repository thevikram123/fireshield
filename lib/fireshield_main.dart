/// Entry point for the FireShield PWA port.
///
///   flutter run -t lib/fireshield_main.dart -d chrome
///   flutter build web -t lib/fireshield_main.dart --release
library;

import 'package:flutter/material.dart';

import 'fireshield/fs_app.dart';

void main() => runApp(const FireShieldApp());
