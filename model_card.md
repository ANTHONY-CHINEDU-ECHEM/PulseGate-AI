# Model card: PulseGateNet

## Overview

<table>
<tr><th>Property</th><th>Value</th></tr>
<tr><td>Task</td><td>Single frame presentation attack detection: is this face a live person or a print, a screen or a paper mask</td></tr>
<tr><td>Architecture</td><td>Two stream convolutional network with residual blocks, 957,785 weights</td></tr>
<tr><td>Inputs</td><td>Context view of 112 by 112 pixels (face with surroundings) and texture view of 96 by 96 pixels (native resolution patch), both standardised per image</td></tr>
<tr><td>Outputs</td><td>Three liveness logits (fused, context stream, texture stream) and six class logits (live and five attack species)</td></tr>
<tr><td>File</td><td><code>models/pulsegate_net.onnx</code>, 3.8 megabytes, ONNX opset 17</td></tr>
<tr><td>Metadata</td><td><code>models/pulsegate_net.json</code>: temperature, thresholds, training settings and history of both stages</td></tr>
<tr><td>Runtime</td><td>ONNX Runtime on CPU, about 9 milliseconds for three patches on one thread</td></tr>
<tr><td>Licence</td><td>MIT</td></tr>
</table>

## Intended use

PulseGateNet is one of four signals in PulseGate AI. It is meant to run inside the liveness engine, which gates it with capture quality checks and combines it with challenges, a depth test and a pulse reading. It is suitable for demonstrations, research and as a starting point for a deployment that will calibrate and fine tune it on its own cameras.

It is not meant to be used alone as an access control, to be used without calibration on a new camera environment, or to make decisions about people beyond "is a live face present".

## Training data

34,024 face crops in six classes. Genuine faces come from 3,755 MUCT photographs of 276 people and from 1,310 frames of two real video clips. Attacks are simulated from the same photographs and frames. See the [data card](data_card.md).

## Training procedure

<table>
<tr><th>Setting</th><th>Stage one</th><th>Stage two</th></tr>
<tr><td>Data</td><td>MUCT plus the first video clip</td><td>The same plus the second video clip</td></tr>
<tr><td>Start</td><td>Random weights</td><td>Result of stage one</td></tr>
<tr><td>Epochs</td><td>12, best at epoch 8</td><td>4, best at epoch 1</td></tr>
<tr><td>Optimiser</td><td>AdamW, learning rate 0.002, cosine schedule, weight decay 0.001</td><td>The same at learning rate 0.0006</td></tr>
<tr><td>Loss</td><td colspan="2">Binary cross entropy on the fused head, plus 0.5 times the same on each stream head, plus 0.3 times cross entropy on the species head, label smoothing 0.03, classes balanced by weight</td></tr>
<tr><td>Regularisation</td><td colspan="2">Dropout 0.3 before fusion, one stream hidden at random in 15 percent of samples, exponential moving average of weights with decay 0.999</td></tr>
<tr><td>Augmentation</td><td colspan="2">Mirror, crop jitter, random patch position, camera appearance in 85 percent of samples, background clutter in 60 percent</td></tr>
<tr><td>Selection</td><td colspan="2">Epoch with the lowest equal error rate on validation data</td></tr>
<tr><td>Calibration</td><td colspan="2">Temperature scaling on validation data, temperature 1.01</td></tr>
<tr><td>Hardware and time</td><td>Two CPU cores, 70 minutes</td><td>Two CPU cores, 24 minutes</td></tr>
</table>

Validation error stopped improving after epoch 8 of stage one while training accuracy kept rising, which is the signature of a model limited by the number of people in its data (193) and not by its size.

## Evaluation

All figures below are on the test partition: 4,437 crops of 42 MUCT people who appear nowhere in training or validation, 1,693 genuine and 2,744 attacks. Thresholds were fixed on validation data.

<table>
<tr><th>Metric</th><th>Value</th><th>95 percent interval</th></tr>
<tr><td>Area under the curve</td><td>0.955</td><td>0.946 to 0.962</td></tr>
<tr><td>Equal error rate</td><td>11.4%</td><td>10.2% to 12.6%</td></tr>
<tr><td>BPCER at the balanced threshold</td><td>9.3%</td><td>7.0% to 11.5%</td></tr>
<tr><td>APCER at the balanced threshold, all attacks pooled</td><td>13.3%</td><td>11.8% to 14.7%</td></tr>
<tr><td>APCER at the balanced threshold, worst species</td><td>30.0%</td><td></td></tr>
<tr><td>ACER, mean of worst species APCER and BPCER</td><td>19.6%</td><td></td></tr>
<tr><td>BPCER when pooled APCER is held at 1 percent</td><td>66.2%</td><td>55.4% to 72.3%</td></tr>
<tr><td>Species named correctly</td><td>71.6%</td><td></td></tr>
<tr><td>Expected calibration error</td><td>0.114</td><td></td></tr>
</table>

Intervals come from 400 bootstrap rounds that resample whole people.

<table>
<tr><th>Attack species</th><th>Test samples</th><th>APCER, balanced threshold</th><th>APCER, forgiving threshold</th><th>APCER, strict threshold</th></tr>
<tr><td>Cut out mask</td><td>560</td><td>1.1%</td><td>3.2%</td><td>0.5%</td></tr>
<tr><td>Replay, monitor</td><td>515</td><td>5.2%</td><td>11.1%</td><td>2.1%</td></tr>
<tr><td>Print, matte</td><td>545</td><td>8.6%</td><td>37.8%</td><td>2.6%</td></tr>
<tr><td>Replay, phone or tablet</td><td>564</td><td>20.9%</td><td>43.8%</td><td>9.4%</td></tr>
<tr><td>Print, glossy</td><td>560</td><td>30.0%</td><td>65.4%</td><td>12.0%</td></tr>
<tr><th>All attacks</th><th>2,744</th><th>13.3%</th><th>32.6%</th><th>5.4%</th></tr>
<tr><th>Genuine rejected (BPCER)</th><th>1,693</th><th>9.3%</th><th>0.8%</th><th>27.4%</th></tr>
</table>

The balanced threshold is 0.73, the forgiving one 0.25, the strict one 0.86.

![Which attack the model thinks it sees](../reports/figures/species_confusion.png)

![Training curves](../reports/figures/training_curves.png)

### Genuine people from other cameras

The model was developed in five stages, each judged on the same real footage of people who are not in the training data. The table gives the share of genuine frames rejected at each model's balanced threshold, with the forgiving threshold in brackets.

<table>
<tr><th>Stage</th><th>Validation equal error rate</th><th>Unseen person, known room, 320 frames</th><th>Unseen people, unseen room, 98 frames</th></tr>
<tr><td>1. Fixed input scaling, light augmentation, MUCT only</td><td>7.8%</td><td>98% (93%)</td><td>100% (100%)</td></tr>
<tr><td>2. Standardised inputs, randomised camera appearance, short run</td><td>11.6%</td><td>79% (8%)</td><td>100% (78%)</td></tr>
<tr><td>3. One real clip as second training domain, short run</td><td>11.5%</td><td>8% (1%)</td><td>99% (81%)</td></tr>
<tr><td>4. Background clutter in training</td><td>12.0%</td><td>1% (0%)</td><td>68% (41%)</td></tr>
<tr><td>5. Shipped: the unseen room added as third training domain</td><td>12.4%</td><td>0% (0%)</td><td>room now in training</td></tr>
</table>

Stages 2 and 3 were stopped after 4 and 5 epochs, stage 4 ran for 12, so their validation error rates are not strictly comparable. Stage 1 is the only one whose validation data is MUCT alone with fixed input scaling. The validation set of stage 5 includes frames from both clips.

The lesson of the table is that validation error went up while usefulness went from none to some. The model that looked best in the laboratory was the one that worked nowhere else.

To reproduce a stage, train with the settings below and place the exported files in `models/experiments` under a name that starts with `ablation`. `python manage.py evaluate_domain_gap` scores every such model.

<table>
<tr><th>Stage</th><th>Overrides for <code>python manage.py train</code></th><th>Dataset</th></tr>
<tr><td>1</td><td><code>crop.input_norm=fixed train.degrade_probability=0 train.clutter_probability=0 train.ema_decay=0 train.dropout=0.2 train.stream_dropout=0.1 train.weight_decay=0.0005 train.epochs=14</code></td><td>MUCT only, before any clip is added</td></tr>
<tr><td>2</td><td><code>train.clutter_probability=0 train.epochs=4</code></td><td>MUCT only</td></tr>
<tr><td>3</td><td><code>train.clutter_probability=0</code>, taken after epoch 5 of the 12 epoch schedule</td><td>MUCT and the first clip</td></tr>
<tr><td>4</td><td>none</td><td>MUCT and the first clip</td></tr>
</table>

### Other evaluations

Robustness to compression, blur, exposure, noise and resolution, error rates by gender, glasses, lighting and camera view, the unseen attack protocol and the classical baseline are reported in the README and stored in `reports/metrics.json` and `reports/unseen.json`.

## Limitations

* **Simulated attacks only.** Performance against real prints, real screens and real masks has not been measured. Every APCER above describes simulated instruments.
* **Camera environments.** The model has seen three. The table above shows what happens in a fourth without calibration.
* **Weak species.** Borderless glossy prints and replays on fine pitch phone screens leave little evidence in a single frame.
* **Frontal faces.** Training poses stay within about 30 degrees. The engine only scores frames within 25 degrees of yaw and 22 of pitch.
* **Face size.** Faces under 90 pixels are not scored. Faces above 256 pixels are reduced first.
* **Degraded capture.** Heavy sensor noise and strong underexposure raise the rejection of genuine users. Strong blur raises the acceptance of attacks.
* **Demographics.** Error rates were compared by the gender recorded in MUCT and by glasses. Age and skin tone are not labelled and were not analysed.
* **Calibration.** The expected calibration error of 0.11 on test data means the score should be read as a ranking and not as a literal probability.

## Ethical considerations

A liveness check that rejects honest people unevenly across groups is a fairness problem as well as a product problem. The slice analysis in this repository is limited by the labels of its source data and by simulated attacks. Any deployment should measure false rejection across the people it actually serves, give users a second attempt and a human fallback, and tell them that a liveness check takes place.

Face images are biometric data. The software stores none. The training data follows the terms of its source: MUCT photographs are downloaded by each user from their authors and never redistributed or shown.
