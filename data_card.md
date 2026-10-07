# Data card

## Summary

<table>
<tr><th>Property</th><th>Value</th></tr>
<tr><td>Name</td><td>PulseGate presentation attack dataset</td></tr>
<tr><td>Size</td><td>34,024 face crops: 13,174 genuine and 20,850 attacks</td></tr>
<tr><td>Classes</td><td>1 genuine class and 5 attack instrument species</td></tr>
<tr><td>Main source</td><td>MUCT face database, 3,755 webcam photographs of 276 people, 29,408 crops</td></tr>
<tr><td>Additional sources</td><td>Two real video clips from another camera, 1,310 frames, 4,616 crops</td></tr>
<tr><td>Attack source</td><td>The same photographs and frames presented through the simulator in <code>pulsegate/data/attacks</code></td></tr>
<tr><td>People in MUCT partitions</td><td>193 for training, 41 for validation and 42 for testing, no overlap</td></tr>
<tr><td>Reproducible</td><td>Yes, every sample has its own random seed derived from <code>dataset.seed</code></td></tr>
<tr><td>Distributed with the repository</td><td>No. It is rebuilt locally with three commands.</td></tr>
</table>

## Composition

<table>
<tr><th>Class</th><th>Family</th><th>Train</th><th>Validation</th><th>Test</th><th>Total</th></tr>
<tr><td>Live</td><td>genuine</td><td>9,400</td><td>2,081</td><td>1,693</td><td>13,174</td></tr>
<tr><td>Print, matte</td><td>print</td><td>2,906</td><td>657</td><td>545</td><td>4,108</td></tr>
<tr><td>Print, glossy</td><td>print</td><td>3,057</td><td>687</td><td>560</td><td>4,304</td></tr>
<tr><td>Replay, phone or tablet</td><td>replay</td><td>3,081</td><td>687</td><td>564</td><td>4,332</td></tr>
<tr><td>Replay, monitor</td><td>replay</td><td>2,689</td><td>605</td><td>515</td><td>3,809</td></tr>
<tr><td>Cut out mask</td><td>mask</td><td>3,055</td><td>682</td><td>560</td><td>4,297</td></tr>
<tr><th>All</th><th></th><th>24,188</th><th>5,399</th><th>4,437</th><th>34,024</th></tr>
</table>

<table>
<tr><th>Source</th><th>Train</th><th>Validation</th><th>Test</th></tr>
<tr><td>MUCT photographs, rendered</td><td>20,548</td><td>4,423</td><td>4,437</td></tr>
<tr><td>Video frames, rendered</td><td>2,593</td><td>714</td><td>0</td></tr>
<tr><td>Video frames, unprocessed genuine crops</td><td>1,047</td><td>262</td><td>0</td></tr>
</table>

The test partition contains MUCT people only. The video clips are split by time, the last fifth of each clip is validation, and the people in them are never used for testing.

Every MUCT photograph yields three genuine presentations, each seen through a different random camera, and one presentation per attack species. 632 of the 30,040 planned renders were dropped because the face tracker could not find a usable face in the rendered frame, mostly monitor replays at very close range where the pixel grid swamps the picture. A real capture pipeline would drop those frames for the same reason.

## The genuine source: MUCT

MUCT was recorded at the University of Cape Town by Stephen Milborrow, John Morkel and Fred Nicolls and published in 2010. Volunteers were photographed by five webcams at once (one frontal, two to one side and two at other heights) under ten lighting set ups, at 480 by 640 pixels. The file names carry the subject number, the lighting set, the camera, the recorded gender and whether the person wears glasses.

MUCT suits this project for three reasons. The photographs come from webcams, the same class of sensor a liveness check runs on. The volunteers were chosen for diversity of age and ethnicity. And the five simultaneous views give real 3D view changes with known geometry, which is exactly what is needed to validate head pose and the depth cue.

<table>
<tr><th>Recorded attribute</th><th>Photographs</th></tr>
<tr><td>Female</td><td>1,910</td></tr>
<tr><td>Male</td><td>1,845</td></tr>
<tr><td>With glasses</td><td>679</td></tr>
<tr><td>Without glasses</td><td>3,076</td></tr>
<tr><td>Per camera view</td><td>751 each for the five views</td></tr>
</table>

### Terms of use

MUCT is free for research use. Its authors ask that the photographs are not reproduced in publicly available documents. This repository honours that request strictly: it contains no MUCT photograph, no crop and no figure showing one. `python manage.py download_data` fetches the archives from the authors' own repository to your machine.

## How attack samples are made

For each attack sample the builder draws a source photograph, a second photograph of a different person from the same partition as the scene behind the medium, a medium with random properties, a placement and a camera. The simulator renders the scene in linear light and the camera model turns it into a frame. The face tracker then locates the face exactly as it would at inference time, and a square of 1.8 times the face box is stored at native resolution.

<table>
<tr><th>Species</th><th>What varies</th></tr>
<tr><td>Print, matte</td><td>Halftone or stochastic printing, screen frequency, ink density and impurity, paper white and black level, saturation, fibre texture, printer banding, border, lighting gradient, shadow</td></tr>
<tr><td>Print, glossy</td><td>Continuous tone or fine grain, deeper blacks, specular gloss patches, border, lighting gradient, shadow</td></tr>
<tr><td>Replay, phone or tablet</td><td>Pixel pitch seen by the camera, stripe order, brightness, black level, colour temperature, glare, veil, bezel width and colour, corner radius, refresh banding, colour fringes</td></tr>
<tr><td>Replay, monitor</td><td>As above with a coarser pitch, which gives stronger moire, and a landscape panel that often fills the frame</td></tr>
<tr><td>Cut out mask</td><td>Print properties, mask size, irregular cut line, eye and mouth holes, visible paper rim, bending shade, shadow on the wearer</td></tr>
<tr><td>All, including live</td><td>Face size from 90 to about 250 pixels (median 159), position, tilt, auto exposure, white balance, lens blur, motion blur, resolution loss, sensor noise, tone curve, saturation, sharpening, JPEG quality</td></tr>
</table>

## Safeguards against shortcuts

A detector trained on simulated attacks will happily learn anything that separates the two classes, including artefacts of the simulation that have nothing to do with real attacks. The dataset is built to close the obvious shortcuts.

* **One camera for everyone.** Genuine and attack frames pass through the same random camera model, so blur, noise, compression and colour balance carry no label information on their own.
* **Auto exposure.** The camera meters the scene, so displays are not systematically brighter and prints not systematically darker than faces.
* **Same generation count.** A genuine sample is a MUCT JPEG resampled once and compressed again. An attack sample is a MUCT JPEG resampled into the medium, resampled by the camera and compressed again. Both have been through at least two rounds of compression.
* **Backgrounds from the same pool.** The scene behind a print or a phone is another MUCT photograph from the same partition, so the room does not reveal the class.
* **Subject disjoint partitions.** No person appears in two partitions, neither as the face nor as the background.
* **Same face localisation.** Crops are cut by the production tracker on the rendered frame, not from known coordinates.

These measures were checked on a trial build by comparing brightness, contrast, saturation and sharpness between classes. After the adjustments the classes overlap on all four, with matte prints remaining somewhat flatter and less saturated, which is true of matte prints.

## Known gaps

* **The attacks are simulated.** No real photograph of a print or a screen is in the data. How well the simulator matches real instruments is the central open question of this project and is not answered by any number computed on this dataset.
* **Three genuine sources, two of them small.** MUCT comes from one room and one set of webcams of an older generation. The two video clips add one more camera, two more rooms and a handful of people. A model trained on three camera environments has not learned what cameras look like in general.
* **Backgrounds are plain.** MUCT was shot against a studio backdrop.
* **Poses are near frontal.** Yaw reaches about 25 degrees in the side cameras. The application only scores near frontal frames for this reason.
* **Demographic labels are limited** to the gender recorded by the MUCT authors and the presence of glasses. Age and skin tone are not labelled, so error rates cannot be broken down by them.
* **No 3D masks, no make up attacks, no deepfakes.**

## Second data source: sample videos

Four clips from the Intel IoT DevKit sample videos collection (CC BY 4.0) show several people filmed at 768 by 432 pixels and 12 frames per second with heavy video compression, in two rooms. They are used in three ways.

<table>
<tr><th>Clip</th><th>Content</th><th>Role</th></tr>
<tr><td>Head pose, man</td><td>One man in front of a plain wall, 134 seconds</td><td>Training domain: 900 frames</td></tr>
<tr><td>Walking and pause</td><td>Several people in a hallway with doors and windows, 91 seconds</td><td>Held out as the unseen room for stages 1 to 4 of the camera comparison, then added as a training domain for the shipped model: 410 frames</td></tr>
<tr><td>Head pose, woman</td><td>One woman in front of the same plain wall, 134 seconds</td><td>Never used for training. Unseen person for every evaluation on real video, source of the sample clip and of every face shown in figures</td></tr>
<tr><td>Head pose, woman and man</td><td>Both people together</td><td>Not used</td></tr>
</table>

Every frame that enters training contributes an unprocessed genuine crop, exactly as the camera recorded it. Every second frame also goes through the simulator, once as a genuine presentation and once per attack species, so that the person and the room appear in every class and cannot become a shortcut.

Why real video is in the training data at all is the main finding of the project: without it the model rejected nearly every real person filmed by a camera other than the MUCT webcams. The README shows the measurements.
