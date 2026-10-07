# Third party material

PulseGate AI is released under the MIT licence. It bundles or downloads the
following material from other parties, each under its own terms.

## Bundled in `models/third_party`

**YuNet face detector** (`face_detection_yunet_2023mar.onnx`)
From the OpenCV Model Zoo (https://github.com/opencv/opencv_zoo), MIT licence. Wu et al., "YuNet: A Tiny Millisecond
level Face Detector", Machine Intelligence Research, 2023.

**MediaPipe Face Mesh** (`face_mesh_468.onnx`, `canonical_face_468.npy`)
From Google MediaPipe release 0.8 (https://github.com/google-ai-edge/mediapipe), Apache Licence 2.0. The network was converted
from TensorFlow Lite to ONNX without changing its weights. The canonical face
array holds the vertex positions of `canonical_face_model.obj`.
Kartynnik et al., "Real time Facial Surface Geometry from Monocular Video on
Mobile GPUs", 2019.

## Bundled in `assets/samples` and shown in figures

**Sample footage** from the Intel IoT DevKit sample videos collection
(https://github.com/intel-iot-devkit/sample-videos), Creative Commons
Attribution 4.0 International (https://creativecommons.org/licenses/by/4.0/).
Changes made: `assets/samples/live_clip.mp4` is a 31 second excerpt that was
encoded again, `assets/samples/face.jpg` is a single frame. Every face shown in
`docs/images` and `reports/figures` comes from this collection, either
unchanged, with an overlay drawn on it, or passed through the attack simulator
of this project.

## Downloaded on demand, never redistributed

**MUCT face database** by Stephen Milborrow, John Morkel and Fred Nicolls,
University of Cape Town, 2010 (https://github.com/StephenMilborrow/muct). Free for research use. The authors ask that the
photographs are not reproduced in public documents. This repository therefore
contains no MUCT photograph, no crop of one and no figure that shows one. The
download command fetches the archives from the authors' own repository.

Two of the sample clips are also used as training data for the shipped model,
as described in `docs/data_card.md`.
