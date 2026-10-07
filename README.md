# Bundled third party networks

<table>
<tr><th>File</th><th>What it is</th><th>Origin</th><th>Licence</th></tr>
<tr><td><code>face_detection_yunet_2023mar.onnx</code></td><td>YuNet face detector</td><td>OpenCV Model Zoo, unchanged</td><td>MIT, see <code>LICENSE_yunet_mit.txt</code></td></tr>
<tr><td><code>face_mesh_468.onnx</code></td><td>MediaPipe Face Mesh, 468 landmarks</td><td>Google MediaPipe release 0.8.6, file <code>face_landmark.tflite</code>, converted from TensorFlow Lite to ONNX with the weights unchanged</td><td>Apache 2.0, see <code>LICENSE_mediapipe_apache_2.0.txt</code></td></tr>
<tr><td><code>canonical_face_468.npy</code></td><td>Vertex positions of the canonical face model</td><td>Google MediaPipe release 0.8.9, file <code>canonical_face_model.obj</code>, vertices only</td><td>Apache 2.0</td></tr>
</table>

Checksums (SHA 256) of the files as shipped are listed in `checksums.txt`.
