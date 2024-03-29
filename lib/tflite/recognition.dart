import 'dart:math';
import 'package:flutter/cupertino.dart';
import 'package:object_detection_sample/camera/camera_view_singleton.dart';

// Represents the recognition output from the model (the square with the score)
class Recognition {

  final int _id;

  final String _label;

  /// Confidence [0.0, 1.0]
  final double _score;

  // Location of bounding box rect
  // The rectangle corresponds to the raw input image
  // passed for inference
  late Rect location;

  Recognition(this._id, this._label, this._score, {required this.location});

  int get id => _id;

  String get label => _label;

  double get score => _score;

  /// Returns bounding box rectangle corresponding to the image on screen
  /// This is the location where the rectangle is rendered on the screen
  Rect get renderLocation {

    double ratioX = CameraViewSingleton.ratio!;
    double ratioY = ratioX;

    double transLeft = max(0.1, location.left * ratioX);
    double transTop = max(0.1, location.top * ratioY);
    double transWidth = min(
        location.width * ratioX, CameraViewSingleton.actualPreviewSize.width);
    double transHeight = min(
        location.height * ratioY, CameraViewSingleton.actualPreviewSize.height);

    Rect transformedRect =
    Rect.fromLTWH(transLeft, transTop, transWidth, transHeight);
    return transformedRect;
  }
}