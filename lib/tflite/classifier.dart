import 'dart:math';
import 'dart:ui';
import 'package:image/image.dart' as image_lib;
import 'package:object_detection_sample/tflite/recognition.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

class Classifier {

  Interpreter? _interpreter;
  Interpreter? get interpreter => _interpreter;

  List<String>? _labels;
  List<String>? get labels => _labels;

  /// Shapes of output tensors
  List<List<int>>? _outputShapes;

  /// Types of output tensors
  List<TfLiteType>? _outputTypes;

  late ImageProcessor imageProcessor;

  static const String MODEL_FILE_NAME = "detect.tflite";
  static const String LABEL_FILE_NAME = "labelmap.txt";

  /// Input size of image (height = width = 300)
  static const int INPUT_SIZE = 300;

  /// the lower limit to recognise an object
  static const double THRESHOLD = 0.5;

  /// Padding the image to transform into square
  late int padSize;

  /// Number of results to show
  static const int NUM_RESULTS = 10;

  Classifier({
    required Interpreter? interpreter,
    required List<String>? labels,
  }) {
    loadModel(interpreter: interpreter);
    loadLabels(labels: labels);
  }

  /// Loads interpreter from asset
  void loadModel({required Interpreter? interpreter}) async {
    try {
      _interpreter = interpreter ??
          await Interpreter.fromAsset(
            MODEL_FILE_NAME,
            options: InterpreterOptions()..threads = 4,
          );

      var outputTensors = _interpreter!.getOutputTensors();
      _outputShapes = [];
      _outputTypes = [];
      for (var tensor in outputTensors) {
        _outputShapes!.add(tensor.shape);
        _outputTypes!.add(tensor.type);
      }
    } catch (e) {
      print("Error while creating interpreter: $e");
    }
  }

  /// Loads labels from assets
  void loadLabels({required List<String>? labels}) async {
    try {
      _labels =
          labels ?? await FileUtil.loadLabels("assets/$LABEL_FILE_NAME");
    } catch (e) {
      print("Error while loading labels: $e");
    }
  }

  /// Runs object detection on the input image
  Map<String, dynamic>? predict(image_lib.Image image) {

    if (_interpreter == null) {
      print("Interpreter not initialized");
      return null;
    }

    // Create TensorImage from image
    TensorImage inputImage = TensorImage.fromImage(image);

    // Pre-process TensorImage
    inputImage = getProcessedImage(inputImage);

    // TensorBuffers for output tensors
    TensorBuffer outputLocations = TensorBufferFloat(_outputShapes![0]);
    TensorBuffer outputClasses = TensorBufferFloat(_outputShapes![1]);
    TensorBuffer outputScores = TensorBufferFloat(_outputShapes![2]);
    TensorBuffer numLocations = TensorBufferFloat(_outputShapes![3]);

    // Inputs object for runForMultipleInputs
    // Use [TensorImage.buffer] or [TensorBuffer.buffer] to pass by reference
    List<Object> inputs = [inputImage.buffer];

    // Outputs map
    Map<int, Object> outputs = {
      0: outputLocations.buffer,
      1: outputClasses.buffer,
      2: outputScores.buffer,
      3: numLocations.buffer,
    };

    // run inference
    _interpreter!.runForMultipleInputs(inputs, outputs);

    // Maximum number of results to show
    int resultsCount = min(NUM_RESULTS, numLocations.getIntValue(0));

    // Using bounding box utils for easy conversion of tensorbuffer to List<Rect>
    List<Rect> locations = BoundingBoxUtils.convert(
      tensor: outputLocations,
      valueIndex: [1, 0, 3, 2],
      boundingBoxAxis: 2,
      boundingBoxType: BoundingBoxType.BOUNDARIES,
      coordinateType: CoordinateType.RATIO,
      height: INPUT_SIZE,
      width: INPUT_SIZE,
    );

    List<Recognition> recognitions = [];

    for (int i = 0; i < resultsCount; i++) {
      // Prediction score
      var score = outputScores.getDoubleValue(i);

      // 1 was added because an object can't have an index of 0
      var labelIndex = outputClasses.getIntValue(i) + 1;
      var label = _labels!.elementAt(labelIndex);

      if (score > THRESHOLD) {
        // inverse of rect
        // [locations] corresponds to the image size 300 X 300
        // inverseTransformRect transforms it our [inputImage]
        Rect transformedRect = imageProcessor.inverseTransformRect(
            locations[i], image.height, image.width);

        recognitions.add(
          Recognition(i, label, score, location: transformedRect),
        );
      }
    }

    return {"recognitions": recognitions};
  }

  /// Pre-process the image
  TensorImage getProcessedImage(TensorImage inputImage) {
    padSize = max(inputImage.height, inputImage.width);

    // create ImageProcessor
    imageProcessor = ImageProcessorBuilder()
    // Padding the image
        .add(ResizeWithCropOrPadOp(padSize, padSize))
    // Resizing to input size
        .add(ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeMethod.BILINEAR))
        .build();

    inputImage = imageProcessor.process(inputImage);
    return inputImage;
  }

  /// another way to predict, didn't try it yet (20/9/2022).
  // Map<String, double>? predictManually(imageLib.Image image) {
  //
  //   if (_interpreter == null) {
  //     print("Interpreter not initialized");
  //     return null;
  //   }
  //
  //   final resizedImage = imageLib.copyResize(image, width: INPUT_SIZE, height: INPUT_SIZE);
  //
  //   var _inputImage = List<List<double>>.generate(INPUT_SIZE, (i) =>
  //       List.generate(INPUT_SIZE, (j) => 0.0)).reshape<double>([1, INPUT_SIZE, INPUT_SIZE, 1]);
  //
  //   for (int x = 0; x < INPUT_SIZE; x++) {
  //     for (int y = 0; y < INPUT_SIZE; y++) {
  //       double val = resizedImage[(x * INPUT_SIZE) + y].toDouble();
  //       val = val > 50 ? 1.0 : 0;
  //       _inputImage[0][x][y][0] = val;
  //     }
  //   }
  //
  //   TensorBuffer outputBuffer = TensorBuffer.createFixedSize(
  //       interpreter!.getOutputTensor(0).shape,
  //       interpreter!.getOutputTensor(0).type);
  //
  //   interpreter!.run(_inputImage, outputBuffer.getBuffer());
  //
  //   final probabilityProcessor = TensorProcessorBuilder()
  //       .add(NormalizeOp(0, 1)).build();
  //
  //   return TensorLabel.fromList(
  //       labels!, probabilityProcessor.process(outputBuffer))
  //       .getMapWithFloatValue();
  // }
}

