# Architecture

![System overview](images/architecture.png)

PulseGate AI turns a stream of camera frames into one of three outcomes: live, not live or inconclusive. Four signals are computed from the same face track and combined by a small, explicit policy. This document explains each stage and names the module that implements it.

## Data flow for one frame

1. **Detection and landmarks** (`pulsegate/vision`). The YuNet detector finds faces. The MediaPipe Face Mesh network predicts 468 landmarks in 3D for the most prominent face. After the first frame the mesh region follows the previous landmarks and the detector only runs every twenty frames, which keeps tracking near four milliseconds per frame on one CPU thread.
2. **Measurements** (`vision/geometry.py`). From the landmarks: head yaw, pitch and roll by fitting the canonical face to the mesh with the Kabsch algorithm, eye aspect ratio for both eyes, mouth opening, eye distance.
3. **Quality gate** (`vision/quality.py`). Face size, position, brightness, sharpness and pose decide whether the frame may be scored, and produce the hint shown to the user when it may not.
4. **Signals** (`pulsegate/signals`, `pulsegate/models`). The four signals below are updated.
5. **Session logic** (`engine/session.py`). A state machine moves through positioning, challenges and analysis, and watches the integrity of the track.
6. **Fusion** (`engine/fusion.py`). When the session ends, the evidence is turned into a verdict with reasons.

## The four signals

### Passive texture model

`models/network.py` defines PulseGateNet, a network of about 0.96 million weights with two streams.

* The **context stream** sees the face with its surroundings (1.8 times the face box) resized to 112 pixels. This is where bezels, paper edges, glare and overall colour are visible.
* The **texture stream** sees a 96 pixel patch from the middle of the face at the native pixel pitch of the camera, never resized. A fixed high pass residual is appended to its input. This is where moire, halftone dots, pixel grids and the noise statistics of a recaptured image are visible.

Each stream has its own classifier, so either can be evaluated alone, and a fusion head combines their features. An auxiliary head names the attack species. The exported ONNX model runs in ONNX Runtime, and three texture patches are averaged per frame.

Three measures keep the network from learning the camera instead of the attack.

* **Standardised inputs.** Every view is shifted and scaled to zero mean and unit spread per colour channel before it reaches the network (`models/preprocess.py`). Exposure, contrast and colour cast are removed by construction.
* **Randomised camera appearance.** During training 85 percent of the crops, genuine and attack alike, get random saturation, tone, blur, resolution loss, noise and compression (`data/dataset.py`).
* **Background clutter.** During training 60 percent of the crops get door frames, wall corners, pictures and lamps drawn in the outer part of the crop, again for every class. Without this the only straight edges the network ever saw near a face were paper borders and bezels, and it flagged real rooms as attacks.

The weights that are validated and exported are an exponential moving average of the training weights, which generalises better than the raw weights on this small set of people.

Scores are calibrated with temperature scaling on validation data. Four operating thresholds are fixed on validation data and stored next to the model in `models/pulsegate_net.json`. Images and passive clips use the balanced one, interactive sessions use a forgiving one, because three more checks stand behind the model there.

### Random challenges

`engine/challenges.py` draws three of five actions (blink twice, turn left, turn right, open the mouth, raise the chin) in random order. Every challenge is judged against the resting pose measured just before its prompt, has to be held for two consecutive frames, has to start after human reaction time and has to finish before a timeout. Turning the wrong way fails at once.

### Depth from motion

`signals/depth.py` collects landmark sets at different head angles and, for every pair at least twelve degrees apart, fits the best plane to plane mapping between them. A real head leaves a residual because the nose moves differently from the cheeks. A flat medium leaves almost none. The residual is measured in eye distances, so it does not depend on image size.

### Remote pulse

`signals/rppg.py` averages the colour of forehead and cheeks in every frame, resamples the series to a uniform rate, applies the plane orthogonal to skin projection, filters to the band of plausible heart rates and looks for a spectral peak. The signal to noise ratio around the peak decides how much the pulse counts. A second series from the background rejects rhythms that come from flickering light.

## Session state machine

<table>
<tr><th>Stage</th><th>What the user sees</th><th>Leaves when</th></tr>
<tr><td>Positioning</td><td>Oval guide and hints such as "Move closer"</td><td>Quality has been good for 1.5 seconds, the resting pose is recorded</td></tr>
<tr><td>Recenter</td><td>"Look straight at the camera"</td><td>The head is frontal and still, and a random pause has passed</td></tr>
<tr><td>Challenge</td><td>The prompt, a progress bar and a countdown</td><td>The action is observed, fails or times out</td></tr>
<tr><td>Hold</td><td>"Hold still"</td><td>Enough frames are scored and the pulse window is filled or three seconds have passed</td></tr>
<tr><td>Done</td><td>Verdict with score and reasons</td><td>The session is final</td></tr>
</table>

In passive mode, used for recorded clips and for flows without prompts, the challenge stages are skipped and spontaneous blinking takes the place of the challenges in the fusion.

## Fusion policy

Hard gates come first. Any of the following ends in not live: an integrity problem, a failed challenge, a face that the depth cue scores as flat, a passive score below half of its threshold. Too few usable frames ends in inconclusive.

Otherwise the available signals are averaged with weights 0.50 for the passive model, 0.25 for challenges, 0.15 for depth and 0.10 for pulse. The passive probability is first rescaled so that the operating threshold of the session mode maps to one half. Signals that could not be measured are left out and the remaining weights are renormalised. The session is live when the result reaches 0.62.

The effect of these numbers is easy to state. A passive score just under its threshold is accepted only when every other check is passed. A score further below additionally needs a clearly detected pulse. A score below half the threshold is never accepted.

<table>
<tr><th>Session mode</th><th>Passive operating point</th><th>Threshold of the shipped model</th><th>Reason</th></tr>
<tr><td>Single image, passive clip</td><td>Balanced: equal error rate on validation data</td><td>0.73</td><td>The model decides almost alone</td></tr>
<tr><td>Interactive</td><td>Forgiving: 1 percent of genuine validation frames rejected</td><td>0.25</td><td>Challenges, depth and pulse cover photographs and recordings</td></tr>
</table>

Both can be changed in `configs/default.yaml` under `passive`, and an explicit `passive.threshold` overrides them.

## Applications

<table>
<tr><th>Module</th><th>Purpose</th></tr>
<tr><td><code>apps/webcam.py</code></td><td>Desktop window with guide, read outs and verdict</td></tr>
<tr><td><code>apps/api.py</code></td><td>HTTP service for images, clips and interactive sessions, plus a browser client</td></tr>
<tr><td><code>apps/hud.py</code></td><td>Overlay drawing shared by the webcam app and the figures</td></tr>
<tr><td><code>cli.py</code></td><td>All commands behind <code>python manage.py</code></td></tr>
</table>

## Training and evaluation pipeline

<table>
<tr><th>Step</th><th>Command</th><th>Module</th><th>Output</th></tr>
<tr><td>Get data</td><td><code>download_data</code></td><td><code>data/muct.py</code></td><td><code>data/raw</code></td></tr>
<tr><td>Render dataset</td><td><code>build_dataset</code></td><td><code>data/builder.py</code>, <code>data/attacks</code></td><td><code>data/processed</code> with a manifest</td></tr>
<tr><td>Add a camera</td><td><code>add_video</code></td><td><code>data/video_domain.py</code></td><td>More rows in the manifest</td></tr>
<tr><td>Train</td><td><code>train</code></td><td><code>training/train.py</code></td><td><code>models/pulsegate_net.onnx</code> and metadata</td></tr>
<tr><td>Test</td><td><code>evaluate</code></td><td><code>evaluation/evaluate.py</code></td><td><code>reports/metrics.json</code></td></tr>
<tr><td>Unseen attacks</td><td><code>evaluate_unseen</code></td><td><code>evaluation/unseen.py</code></td><td><code>reports/unseen.json</code></td></tr>
<tr><td>Unseen cameras</td><td><code>evaluate_domain_gap</code></td><td><code>evaluation/domain_gap.py</code></td><td><code>reports/domain_gap.json</code></td></tr>
<tr><td>Real video</td><td><code>evaluate_sessions</code></td><td><code>evaluation/sessions.py</code></td><td><code>reports/sessions.json</code></td></tr>
<tr><td>Signals</td><td><code>validate_signals</code></td><td><code>evaluation/signals.py</code></td><td><code>reports/signals.json</code></td></tr>
<tr><td>Figures</td><td><code>figures</code></td><td><code>evaluation/plots.py</code></td><td><code>reports/figures</code>, <code>docs/images</code></td></tr>
</table>

## The attack simulator

`pulsegate/data/attacks` renders what a camera would see when a face is shown on paper or on a display. It works in linear light and models the physics that create the cues a detector relies on.

* **Displays** are rendered at subpixel resolution with red, green and blue stripes and dark gaps between rows. The panel is viewed through a tilted plane, integrated over the aperture of each sensor pixel and then sampled. Moire and colour fringes appear by themselves when the display grid and the sensor grid beat against each other. Black level, colour temperature, glare on the cover glass, bezels and refresh banding are added.
* **Prints** are rendered as reflectance maps. Halftone printing uses four rotated ink screens with under colour removal and imperfect inks, stochastic printing adds grain, photo paper adds gloss. Paper fibre, printer banding, white borders, uneven lighting and cast shadows follow.
* **Cut out masks** align a printed face with another person's face, cut it along an irregular oval, optionally open the eyes and mouth, and bend it around the head.
* **The camera** is the same for genuine and attack frames: auto exposure, white balance, lens blur, motion blur, sensor noise, tone curve, sharpening, saturation, resolution loss and JPEG compression, all drawn from one distribution. No class can be recognised by blur, noise or compression alone.

The design choice behind the last point matters most. If attacks were always a little blurrier or a little more compressed than genuine images, a network would learn that shortcut and fail on the first sharp attack or the first blurry webcam.
