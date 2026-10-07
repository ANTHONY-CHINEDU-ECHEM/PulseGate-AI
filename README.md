# PulseGate AI

**Multi Signal Face Liveness Detection for Camera Based Identity Checks**

Presentation attack detection through texture analysis, random challenges, depth from motion and remote pulse measurement, with a real time webcam application, an HTTP service and a fully reproducible training and evaluation pipeline.

![A complete liveness session on real footage](docs/images/session_screens.png)

## Project Brief

Opening a bank account, recovering a locked wallet, collecting a parcel on credit, sitting a remote exam: all of these now begin with a camera instead of a counter clerk. The applicant shows a face, software compares it with a document or a stored template, and access is granted. The comparison step has become very accurate. The weak point sits in front of it. A face matcher cannot tell a person from a picture of that person, so the cheapest attack on a remote identity check is not a forged passport but a photograph held up to the lens, a video played on a phone, or a printed mask with the eyes cut out. Standards bodies call these presentation attacks, and the control that stops them is liveness detection: proof that a real, present human being is in front of the camera at the moment of the check.

Most open implementations of liveness detection follow one recipe. A single image classifier is trained on a single public dataset and reported at 99 percent accuracy or better. Two things go wrong when such a model leaves the laboratory. First, it tends to learn the camera, the lighting and the backdrop of its training data instead of the difference between skin and paper, so it rejects honest users filmed anywhere else. Second, one classifier is asked to cover attacks that have nothing in common. A printed photograph, a recorded video and an accomplice relayed live through a tablet each fail for a different physical reason, and no single cue catches all three.

PulseGate AI is built around that observation. It reads four independent signals from the same face track and lets each one do the job it is suited for. A compact neural network inspects the texture and surroundings of the face for the marks of paper and screens. A random sequence of prompts (blink twice, turn left, raise the chin) checks that the face responds in real time, which a photograph or a recording cannot do. A geometric test checks that a turning head moves like a three dimensional object and not like a tilted sheet. A remote pulse reading looks for a heartbeat in the colour of the skin. An explicit, auditable policy combines the four into one verdict with reasons.

The project was built under a constraint that shaped it. The established datasets of real attack recordings are licensed for request only and could not be used. Instead of quietly training on whatever could be scraped, the project builds a physically based simulator that renders prints, displays and paper masks from genuine face photographs, and it is strict about what that does and does not prove. Everything that could be validated on real data was: head pose and the depth test on real photographs taken by five cameras at once, challenges, blinking and pulse on real video, genuine acceptance on real footage of people the model never saw. What could not be validated on real data, above all the accuracy of the texture model against real printed photographs and real screens, is stated as an open question, and the repository ships the tools to answer it on any camera in a few minutes.

The most useful result of the project is a failure that was caught and explained. The first version of the texture model scored well on its own test set and rejected 98 to 100 percent of real people as soon as the camera changed. The sections below show why, what fixed it, and what it means for anyone who buys or builds a liveness check.

## Key Findings

**1. A model can pass its test set and still reject every real customer.** The first texture model reached an equal error rate of 7.8 percent on validation data. Shown genuine people filmed by a different camera, it rejected 98 percent of frames in one clip and 100 percent in another. It had learned that "live" means the colour, contrast and plain backdrop of the training photographs. Four changes, each measured on the same real footage, brought rejection down step by step.

<table>
<tr><th>Stage of the model</th><th>What changed</th><th>Genuine frames rejected, unseen person in a known room</th><th>Genuine frames rejected, unseen people in an unseen room</th></tr>
<tr><td>1</td><td>First model</td><td>98%</td><td>100%</td></tr>
<tr><td>2</td><td>Every input image is standardised, and training randomises tone, colour, blur, noise and compression for all classes</td><td>79%</td><td>100%</td></tr>
<tr><td>3</td><td>Two minutes of genuine footage from a second camera join the training data</td><td>8%</td><td>99%</td></tr>
<tr><td>4</td><td>Door frames, pictures and lamps are drawn around the head during training</td><td>1%</td><td>68%</td></tr>
<tr><td>5, shipped</td><td>The unseen room joins the training data as a third camera</td><td>0%</td><td>not applicable, now a training room</td></tr>
</table>

Every model is judged at its own balanced threshold from validation data. The first column rests on 320 frames of one person, the second on 98 frames of four people, so the percentages are coarse. The pattern is not. Each stage removed one shortcut: camera style, then codec softness, then the assumption that straight edges near a face are paper borders.

![Genuine people rejected at each stage](reports/figures/domain_gap.png)

**2. The honest accuracy of single frame texture analysis is moderate, and it varies a lot by attack.** On 4,437 test images of 42 people who appear nowhere in training, the shipped model reaches an area under the curve of 0.955 (95 percent interval 0.946 to 0.962) and an equal error rate of 11.4 percent (10.2 to 12.6). At its balanced threshold it rejects 9.3 percent of genuine presentations and accepts 13.3 percent of attacks. Behind that average sit very different instruments.

<table>
<tr><th>Attack instrument</th><th>Accepted at the balanced threshold</th><th>Why</th></tr>
<tr><td>Cut out paper mask</td><td>1.1%</td><td>The cut edge and the flat paper against a real head are visible in every frame</td></tr>
<tr><td>Replay on a monitor</td><td>5.2%</td><td>Coarse pixels beat against the sensor grid and leave moire</td></tr>
<tr><td>Print on matte paper</td><td>8.6%</td><td>Halftone dots, paper fibre and weak blacks</td></tr>
<tr><td>Replay on a phone or tablet</td><td>20.9%</td><td>Fine pixels leave no texture, only the bezel and glare give it away</td></tr>
<tr><td>Print on glossy photo paper</td><td>30.0%</td><td>A borderless photo lab print has almost no single frame cue left</td></tr>
</table>

The numbers are lower than those usually quoted for this task, for a reason that is deliberate. Stage 1 above scored 7.8 percent equal error on validation data by using cues that do not survive a change of camera. Removing those cues cost four points of laboratory accuracy and bought a model that accepts real people. A classical baseline of colour texture descriptors with logistic regression, run through the same protocol, reaches 15.2 percent equal error, so the network earns its place but not by a wide margin.

**3. A detector only knows the attacks it was shown.** One attack family at a time was removed from training and the model was tested on exactly that family. Acceptance of unseen prints and unseen replays roughly doubled (for example from 21.3 to 41.0 percent for phone replays). The paper mask, the easiest instrument when the model has seen it at 1.6 percent, became the hardest when it had not, at 73.6 percent. No amount of training data closes this gap in general, which is the strongest argument for not relying on appearance alone.

**4. A recording cannot follow random prompts.** The challenge protocol was run 592 times against real recorded footage of people turning, nodding, blinking and opening their mouths, which is the best material an attacker with a recording could hope for. Each simulated session started at a random point of a clip and drew its own three prompts. None of the 592 passed. The first prompt alone was satisfied by coincidence in 0 to 14 percent of cases, depending on the action, and three in a row never.

**5. A tilted photograph does not look like a turning head, and the difference is measurable.** The depth test was validated on real photographs: 276 people photographed by five cameras at the same instant, which gives true changes of viewpoint. Against 552 software tilted copies of the frontal photographs it separated 433 real view pairs perfectly, with an area under the curve of 1.000. A second result came for free. Tilting a flat photograph by 15 to 30 degrees moved the estimated head yaw by a median of only 2.8 degrees, and no tilted photograph reached the 16 degrees a turn prompt requires. Faking a head turn with a sheet of paper fails twice.

![Depth from motion on real multi camera photographs](reports/figures/depth_cue.png)

**6. Remote pulse works when the signal is there and is honest about it when it is not.** On heavily compressed real video at 12 frames per second the skin signal was too weak to carry a pulse reading on its own (median signal to noise ratio of minus 3.2 decibel). When a synthetic heartbeat of known rate was injected into the skin pixels of that same footage at 0.6 percent amplitude, the whole pipeline recovered the rate within 3 beats per minute in 12 of 12 runs, with a mean error of 0.3 beats. Pulse is therefore wired in as supporting evidence only: it can lift a borderline session and can never sink a real user.

**7. On real video of a person the model never saw, sessions came out right.** Thirteen ten second windows of genuine footage were all accepted. Twenty attack sessions made by presenting the same footage through the simulator, with one fixed medium per session and hand tremor, were all turned away (19 rejected, 1 inconclusive). The sample is small and the attacks are simulated, so this is a consistency check, not a security certificate.

**8. It runs in real time on one CPU thread.** A full session update, covering tracking, 468 landmarks, the texture model on three patches, blink, depth and pulse bookkeeping, takes 14 milliseconds at the median and 22 milliseconds at the 95th percentile on a two core cloud CPU. No GPU is needed for inference. The texture model has 957,785 weights and is a 3.8 megabyte file.

**What this means for a deployment.** Liveness is a layered control, not a classifier. The layers that rest on geometry and timing (challenges, depth) were validated on real data and are strong against the cheap attacks that make up most fraud attempts. The layer that rests on appearance (the texture model) is the only one that sees a screen relaying a live accomplice, and it is also the one that must be calibrated for each camera environment. Any vendor claim of a single accuracy figure for liveness should be met with two questions: on whose cameras, and against which attacks.

## Picture Examples

**What the camera sees.** One genuine presentation and the five attack instruments the simulator produces, with a magnified skin patch below each. The patches show why the texture stream works at native resolution: halftone dots and moire live at the scale of single pixels and vanish when an image is resized.

![Genuine presentation and five attack instruments](docs/images/attack_gallery.png)

**What the model reacts to.** Grad CAM maps for the context stream, drawn on one brightness scale. For the phone replay the evidence sits on the bezel. The monitor replay in this row fills the frame, shows no edge and is accepted with a score of 0.91, which is exactly the kind of attack the other three signals exist for.

![Where the model finds evidence of an attack](docs/images/explanations.png)

**Blink detection on real footage.** Eye opening relative to the user's own baseline over thirty seconds. Of 17 events detected in the full clip, 16 were confirmed as blinks by inspecting the frames and 1 was a long downward glance.

![Blink detection](reports/figures/blink_trace.png)

**Remote pulse.** The skin colour signal and its spectrum for real footage as recorded, and for the same footage with an injected heartbeat of 84 beats per minute.

![Remote pulse](reports/figures/pulse.png)

All faces in these figures come from openly licensed sample footage. No photograph from the training database is shown anywhere in this repository, as its authors request.

## How It Works

![Architecture](docs/images/architecture.png)

<table>
<tr><th>Signal</th><th>What it measures</th><th>Attack it stops</th><th>Validated on</th></tr>
<tr><td>Passive texture model</td><td>Paper, pixel grids, bezels and glare in a single frame, with a two stream network called PulseGateNet</td><td>Prints and screens, including a screen that relays a live accomplice</td><td>Simulated attacks on 42 unseen people, genuine footage of people and rooms held out during development</td></tr>
<tr><td>Random challenges</td><td>Three of five actions in random order with random timing</td><td>Photographs and recorded video</td><td>592 sessions against real recorded footage</td></tr>
<tr><td>Depth from motion</td><td>Whether landmark motion between two head poses can be explained by a flat surface</td><td>Prints and phones tilted to fake a turn, paper masks</td><td>Real photographs of 276 people from five simultaneous cameras</td></tr>
<tr><td>Remote pulse</td><td>Heartbeat in the colour of forehead and cheeks</td><td>Adds confidence, never decides alone</td><td>Real video with an injected heartbeat of known rate</td></tr>
</table>

A session moves through positioning, three challenges with a return to rest between them, and a short hold. Throughout, the engine watches that the same face stays in view: a face that disappears, jumps or is joined by a second face ends the session.

The verdict is one of live, not live or inconclusive, and always carries reasons. Hard gates come first: a failed challenge, a face that moves like a flat surface, or a texture score below half of its threshold is disqualifying whatever the other signals say. What remains is a weighted average (texture 0.50, challenges 0.25, depth 0.15, pulse 0.10) in which signals that could not be measured are left out instead of counted as zero. Poor light or a session that ran out of time gives inconclusive, so that an honest user is asked to try again instead of being locked out.

Two operating points are used for the texture model. When it decides almost alone, on a still image or a passive clip, it runs at its balanced threshold. In an interactive session three more checks stand behind it, so it runs at a forgiving threshold that rejects under 1 percent of genuine frames and still turns away two thirds of simulated attacks on its own.

Design details are in [docs/architecture.md](docs/architecture.md), the attacker model in [docs/threat_model.md](docs/threat_model.md).

## Data

<table>
<tr><th>Property</th><th>Value</th></tr>
<tr><td>Face crops</td><td>34,024, of which 13,174 genuine and 20,850 attacks</td></tr>
<tr><td>Classes</td><td>Live, print on matte paper, print on glossy paper, replay on phone or tablet, replay on monitor, cut out mask</td></tr>
<tr><td>Main source</td><td>MUCT face database: 3,755 webcam photographs of 276 people, five cameras, ten lighting set ups</td></tr>
<tr><td>Second and third source</td><td>1,310 frames from two openly licensed video clips, used unprocessed as genuine samples and as source for simulated attacks</td></tr>
<tr><td>Partitions</td><td>24,188 for training, 5,399 for validation, 4,437 for testing</td></tr>
<tr><td>People</td><td>193 in training, 41 in validation, 42 in testing, no overlap</td></tr>
</table>

Attack samples are produced by a simulator that models the physics behind the cues a detector relies on. Displays are rendered at subpixel resolution, viewed through a tilted plane and integrated over the aperture of each sensor pixel, so moire appears by itself where a display grid and a sensor grid beat against each other. Prints are rendered as reflectance maps with four rotated halftone screens, under colour removal, imperfect inks, paper fibre and gloss. Paper masks are aligned to another person's face, cut along an irregular line and bent around the head.

Genuine and attack frames then pass through one shared camera model with auto exposure, white balance, lens and motion blur, sensor noise, sharpening, resolution loss and compression. This matters more than any detail of the simulator. If attacks were always slightly blurrier or more compressed than genuine frames, a network would learn that instead of learning the attack.

The full description, including terms of use and known gaps, is in [docs/data_card.md](docs/data_card.md).

## Results in Detail

### Texture model on the test partition

<table>
<tr><th>Model</th><th>Area under the curve</th><th>Equal error rate</th></tr>
<tr><td>PulseGateNet, both streams</td><td>0.955</td><td>11.4%</td></tr>
<tr><td>Context stream alone</td><td>0.945</td><td>12.5%</td></tr>
<tr><td>Texture stream alone</td><td>0.867</td><td>22.6%</td></tr>
<tr><td>Classical colour texture baseline</td><td>0.928</td><td>15.2%</td></tr>
</table>

<table>
<tr><th>Operating point, fixed on validation data</th><th>Threshold</th><th>Genuine rejected (BPCER)</th><th>Attacks accepted (APCER)</th><th>Worst instrument</th></tr>
<tr><td>Forgiving, used in interactive sessions</td><td>0.25</td><td>0.8%</td><td>32.6%</td><td>65.4%</td></tr>
<tr><td>Balanced, used for images and passive clips</td><td>0.73</td><td>9.3%</td><td>13.3%</td><td>30.0%</td></tr>
<tr><td>Strict, 5 percent attack acceptance</td><td>0.86</td><td>27.4%</td><td>5.4%</td><td>12.0%</td></tr>
<tr><td>Very strict, 1 percent attack acceptance</td><td>0.95</td><td>69.0%</td><td>0.8%</td><td>1.8%</td></tr>
</table>

Error rates follow ISO/IEC 30107 part 3. APCER is the share of attacks accepted, reported for all attacks pooled and for the worst instrument. BPCER is the share of genuine presentations rejected. Thresholds were chosen on validation data and never adjusted on the test partition. Confidence intervals come from resampling whole people, not single images.

![Score distribution](reports/figures/score_distribution.png)

![Detection error trade off](reports/figures/det_curves.png)

### Attacks the model was never shown

To see what happens with an attack nobody planned for, one attack family at a time was removed from training and validation, and the resulting model was tested on exactly that family. All runs, including the reference run that sees every family, use the same reduced training budget.

<table>
<tr><th>Attack instrument</th><th>Accepted when its family was in training</th><th>Accepted when its family was never seen</th></tr>
<tr><td>Print on matte paper</td><td>8.8%</td><td>18.9%</td></tr>
<tr><td>Print on glossy photo paper</td><td>30.0%</td><td>44.1%</td></tr>
<tr><td>Replay on a phone or tablet</td><td>21.3%</td><td>41.0%</td></tr>
<tr><td>Replay on a monitor</td><td>5.8%</td><td>13.0%</td></tr>
<tr><td>Cut out paper mask</td><td>1.6%</td><td>73.6%</td></tr>
</table>

Each run trains for 5 epochs on 14,000 samples and is judged at its balanced threshold from validation data that excludes the held out family. Genuine rejection was 14.9 percent in the reference run and 9.0, 11.3 and 12.9 percent in the runs without prints, replays and masks.

Prints and replays roughly double their acceptance when the model has never seen their family, so part of what it learned carries over. The paper mask is the opposite case: trivial once seen, and accepted three times out of four when not. A mask shows real hair, real ears and a real room, and only examples teach a network to look at the cut line. The practical reading is that an attack catalogue is never finished, and that the checks which do not depend on appearance are what cover the gap. A paper mask that fools the texture model still fails the depth test.

![Attack acceptance by instrument](reports/figures/apcer_by_species.png)

### Fairness across capture conditions

Genuine presentations rejected at the balanced threshold, by the attributes recorded in the source database.

<table>
<tr><th>Group</th><th>Genuine rejected</th><th>Group</th><th>Genuine rejected</th></tr>
<tr><td>Recorded as female</td><td>8.8%</td><td>Without glasses</td><td>8.8%</td></tr>
<tr><td>Recorded as male</td><td>9.9%</td><td>With glasses</td><td>12.5%</td></tr>
<tr><td>Best lighting set</td><td>3.3%</td><td>Frontal camera</td><td>7.4%</td></tr>
<tr><td>Worst lighting set</td><td>14.7%</td><td>Side camera at 23 degrees</td><td>12.8%</td></tr>
</table>

The gap between the recorded genders is small. Glasses and lighting matter more. Age and skin tone are not labelled in the source database, so no claim is made about them. A deployment should measure both on its own users.

![Genuine rejection by condition](reports/figures/bpcer_by_condition.png)

### Robustness to image quality

The test images were degraded one property at a time with the threshold left unchanged. Compression down to JPEG quality 15 and resolution down to 30 percent barely move the error rates. Strong blur lets more attacks through, because it erases their texture. Heavy sensor noise and severe underexposure raise the rejection of genuine users, which is why the application checks brightness and sharpness before it scores a frame.

![Robustness](reports/figures/robustness.png)

### Signals validated on real data

<table>
<tr><th>Check</th><th>Data</th><th>Result</th></tr>
<tr><td>Head pose</td><td>276 people, three cameras at known horizontal offsets</td><td>Mean yaw of 3.0, 11.8 and 23.4 degrees with a spread of about 4 degrees per camera</td></tr>
<tr><td>Depth test</td><td>433 real view pairs against 552 tilted photographs</td><td>Area under the curve 1.000, 99.8% of real pairs above the threshold, none of the flat ones</td></tr>
<tr><td>Faking a turn with a flat photograph</td><td>552 photographs tilted by 15 to 30 degrees</td><td>Estimated yaw moved by a median of 2.8 degrees, none reached the 16 degree threshold</td></tr>
<tr><td>Random challenges against recordings</td><td>592 sessions on two clips of recorded head movement</td><td>0 sessions passed all three prompts</td></tr>
<tr><td>Blink detection</td><td>One clip of 134 seconds</td><td>17 events detected, 16 confirmed by eye, missed blinks not counted</td></tr>
<tr><td>Pulse, footage as recorded</td><td>Four windows of 12 seconds</td><td>Median signal to noise ratio of minus 3.2 decibel, no reliable reading</td></tr>
<tr><td>Pulse, injected at 0.3% amplitude</td><td>12 runs at 66, 84 and 102 beats per minute</td><td>9 of 12 within 3 beats per minute</td></tr>
<tr><td>Pulse, injected at 0.6% amplitude</td><td>12 runs</td><td>12 of 12 within 3 beats per minute, mean error 0.3</td></tr>
<tr><td>Full sessions, genuine</td><td>13 windows of real video, unseen person</td><td>13 accepted</td></tr>
<tr><td>Full sessions, simulated attacks</td><td>20 windows, four per instrument</td><td>0 accepted, 19 rejected, 1 inconclusive</td></tr>
</table>

![Sessions on real video](reports/figures/video_sessions.png)

### Latency

<table>
<tr><th>Stage</th><th>Median</th><th>95th percentile</th></tr>
<tr><td>Face detection, runs every twentieth frame</td><td>8.9 ms</td><td>10.5 ms</td></tr>
<tr><td>Tracking with 468 landmarks</td><td>4.2 ms</td><td>17.4 ms</td></tr>
<tr><td>Texture model, three patches</td><td>9.1 ms</td><td>11.9 ms</td></tr>
<tr><td>Full session update</td><td>14.1 ms</td><td>21.6 ms</td></tr>
</table>

Measured on 768 by 432 frames with one inference thread on a two core cloud CPU. Run `python manage.py benchmark` for your own machine.

## Limitations and Responsible Use

* **The attacks in training and testing are simulated.** No real photograph of a print or a screen was available. The error rates of the texture model against real attack instruments are unknown. The simulator is physically motivated and its shortcuts were closed with care, but that is an argument, not a measurement.
* **The texture model does not transfer to a new room by itself.** Before the hallway clip was added to training, 68 percent of genuine frames from that room were rejected at the balanced threshold and 41 percent at the forgiving one. The shipped model has seen three camera environments. Yours is a fourth. Use the calibration commands below before trusting it.
* **Glossy prints and phone replays are weak spots** for single frame analysis, at 30 and 21 percent acceptance. In an interactive session they still have to pass the challenges and the depth test, which a photograph cannot. A live accomplice relayed on a phone has only the texture model to beat.
* **Evaluation on real video is small**: one unseen person in a known room for sessions, four people for the unseen room.
* **Out of scope**: silicone and 3D printed masks, deepfakes injected into the video stream, virtual cameras. These need sensors or platform controls that pixels alone cannot replace.
* **Liveness is not identity.** Pair this with face matching and bind both to the same session.
* **Faces are biometric data.** The software stores no images. Session results are JSON without pictures. If you keep images for audit, you take on the legal duties that come with them.

This is a portfolio and research project. It has not been certified against ISO/IEC 30107 and should not guard real accounts without evaluation on real attack instruments.

## Getting Started

Python 3.10 or newer. Clone the repository, then from its root:

```
pip install .
python manage.py webcam
```

That is enough to run the live check with the shipped model. A window opens with an oval guide, asks for three actions and shows the verdict with the score and the reasons. Keys: `r` restarts, `p` switches to passive mode, `i` back to interactive, `s` saves the session record, `q` quits.

Other ways to run it:

```
python manage.py analyze video=assets/samples/live_clip.mp4
python manage.py analyze image=assets/samples/face.jpg
python manage.py serve
python manage.py benchmark
python manage.py info
```

`serve` starts the HTTP service on port 8000 with a browser client at the root address. With Docker, `docker compose up` does the same in a container that holds the inference path only. The interface is documented in [docs/api.md](docs/api.md).

Every option lives in `configs/default.yaml` and can be overridden on the command line as `section.key=value`:

```
python manage.py webcam webcam.camera_index=1 challenge.count=2
python manage.py serve api.port=9000
```

## Calibrating on Your Own Camera

The single most valuable thing to do with this repository is to measure it on your own camera and your own attacks. It takes about ten minutes.

```
python manage.py collect label=live
python manage.py collect label=replay_phone
python manage.py collect label=print_matte
python manage.py calibrate
```

`collect` records twenty seconds of face crops under the given label. Sit normally for `live`. For the attack labels hold a phone showing a face, or a printed photograph, in front of the camera. `calibrate` then reports how many of your genuine frames the shipped model rejects and how many of your attacks it accepts. `python manage.py calibrate apply=true` stores a threshold fitted to your captures in `configs/local.yaml`.

To go further and adapt the model itself:

```
python manage.py add_video video=my_clip.mp4 name=my_room
python manage.py finetune
```

`add_video` adds a clip of genuine footage as a new training domain, including simulated attacks rendered from it. This is the step that took rejection of real people from 99 to 0 percent in the stages above. `finetune` continues training on your labelled captures mixed with the original data, refits the thresholds and exports a new model.

## Reproducing the Results

```
pip install ".[train,dev]"
python manage.py reproduce
```

`reproduce` runs the whole pipeline in order. The steps can also be run one at a time.

<table>
<tr><th>Command</th><th>What it does</th><th>Time on two CPU cores</th></tr>
<tr><td><code>download_data</code></td><td>Fetches MUCT from its authors' repository and four sample clips</td><td>2 min</td></tr>
<tr><td><code>build_dataset</code></td><td>Renders 30,040 presentations from MUCT</td><td>72 min</td></tr>
<tr><td><code>add_video</code></td><td>Adds a real clip as a training domain</td><td>8 min per clip</td></tr>
<tr><td><code>train</code></td><td>Trains, calibrates and exports the texture model</td><td>70 min for 12 epochs</td></tr>
<tr><td><code>evaluate</code></td><td>Test metrics, baseline, robustness, slices</td><td>9 min</td></tr>
<tr><td><code>evaluate_unseen</code></td><td>Unseen attack protocol, four short training runs</td><td>70 min</td></tr>
<tr><td><code>evaluate_domain_gap</code></td><td>Genuine footage from unseen cameras</td><td>1 min</td></tr>
<tr><td><code>evaluate_sessions</code></td><td>Genuine and simulated attack sessions on real video</td><td>6 min</td></tr>
<tr><td><code>validate_signals</code></td><td>Depth, challenges, blinks and pulse on real data</td><td>6 min</td></tr>
<tr><td><code>figures</code></td><td>Rebuilds every figure from the report files</td><td>2 min</td></tr>
</table>

The shipped model was trained in two stages, both recorded in `models/pulsegate_net.json`: 12 epochs on MUCT plus the first clip, then 4 epochs at a lower learning rate after the second clip was added, starting from the first result.

```
python manage.py add_video
python manage.py train train.run_name=pulsegate_stage1
python manage.py add_video video=data/raw/videos/face_demographics_walking_and_pause.mp4 name=intel_hallway max_frames=600
python manage.py train train.init_checkpoint=models/pulsegate_stage1.pt train.epochs=4 train.lr=0.0006 train.warmup_epochs=0.3
```

Stage one is also stage 4 of the camera comparison in the key findings. Training is resumable: an interrupted run continues from its last finished epoch. Every random choice is seeded, although results can differ slightly between machines and library versions.

## Repository Structure

```
pulsegate_ai/
├── manage.py                    command line entry point
├── configs/default.yaml         every setting in one place
├── pulsegate/
│   ├── vision/                  face detection, 468 landmarks, head pose, capture quality
│   ├── signals/                 blink, depth from motion, remote pulse
│   ├── models/                  PulseGateNet, preprocessing, ONNX inference, Grad CAM, baseline
│   ├── engine/                  challenges, session state machine, fusion policy
│   ├── data/                    MUCT access, dataset builder, video domains
│   │   └── attacks/             the presentation attack simulator
│   ├── training/                training loop, fine tuning
│   ├── evaluation/              metrics, protocols, signal validation, figures
│   └── apps/                    webcam app, HTTP service, browser client
├── models/                      shipped texture model and bundled third party networks
├── assets/samples/              a clip and a still for demos and tests
├── reports/                     metrics as JSON and CSV, figures
├── docs/                        architecture, threat model, data card, model card, API
└── tests/                       unit and integration tests
```

## Tests

```
pytest
```

A suite of 156 tests covers the metrics against hand computed cases and scikit learn, geometry against known rotations, every signal on synthetic input with a known answer, the simulator, the network and its ONNX export, and the engine and HTTP service end to end on real footage. The suite also passes without PyTorch installed, which is how the inference container runs and where the training tests are skipped, and on OpenCV 4.10 as well as 5.0.

## Documents

* [Architecture](docs/architecture.md): how the pieces fit and why
* [Threat model](docs/threat_model.md): which attacks are in scope and which signal stops which
* [Data card](docs/data_card.md): sources, composition, safeguards, gaps
* [Model card](docs/model_card.md): intended use, metrics, caveats
* [HTTP service](docs/api.md): endpoints, fields, reason codes

## Licence and Acknowledgements

The code is released under the MIT licence. Third party material keeps its own terms, listed in [NOTICE.md](NOTICE.md).

* **MUCT face database** by Stephen Milborrow, John Morkel and Fred Nicolls, University of Cape Town. Free for research use. Downloaded on demand, never redistributed, never shown.
* **YuNet face detector** from the OpenCV Model Zoo, MIT licence.
* **MediaPipe Face Mesh** by Google, Apache Licence 2.0, converted to ONNX.
* **Sample footage** from the Intel IoT DevKit sample videos collection, CC BY 4.0.

## References

* ISO/IEC 30107 part 3: Biometric presentation attack detection, testing and reporting.
* Wang, den Brinker, Stuijk and de Haan (2017). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering.
* de Haan and Jeanne (2013). Robust pulse rate from chrominance based rPPG. IEEE Transactions on Biomedical Engineering.
* Soukupova and Cech (2016). Real time eye blink detection using facial landmarks. Computer Vision Winter Workshop.
* Boulkenafet, Komulainen and Hadid (2016). Face spoofing detection using colour texture analysis. IEEE Transactions on Information Forensics and Security.
* Kartynnik, Ablavatski, Grishchenko and Grundmann (2019). Real time facial surface geometry from monocular video on mobile GPUs.
* Wu, Peng and Yu (2023). YuNet: a tiny millisecond level face detector. Machine Intelligence Research.
* Guo, Pleiss, Sun and Weinberger (2017). On calibration of modern neural networks. International Conference on Machine Learning.
* Milborrow, Morkel and Nicolls (2010). The MUCT landmarked face database. Pattern Recognition Association of South Africa.
