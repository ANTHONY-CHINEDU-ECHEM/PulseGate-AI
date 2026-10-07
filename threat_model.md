# Threat model

This document states what PulseGate AI is meant to stop, what it is not meant to stop, and which part of the system is responsible for each attack. It follows the vocabulary of ISO/IEC 30107, where an attempt to fool a biometric capture device is a presentation attack and the object used is a presentation attack instrument.

## What is protected

A camera based check that a real, present person is in front of the device at the time of the check. Typical uses are remote customer onboarding, account recovery, step up authentication before a sensitive action, and attendance or exam proctoring.

Liveness is not identity. PulseGate AI answers "is this a live human face" and nothing else. A complete identity flow pairs it with face matching against a trusted reference, and binds both to the same capture session so that a live person cannot pass the liveness step and then hand over to a photograph for the matching step.

## Attacker capabilities considered

<table>
<tr><th>Level</th><th>Attacker has</th><th>Example</th><th>In scope</th></tr>
<tr><td>A</td><td>A still image of the victim and everyday equipment</td><td>Printed photograph, photograph on a phone</td><td>Yes</td></tr>
<tr><td>A</td><td>A video of the victim and everyday equipment</td><td>Social media clip replayed on a tablet or monitor</td><td>Yes</td></tr>
<tr><td>B</td><td>A print and some craft</td><td>Paper mask with the eyes cut out, worn by the attacker</td><td>Yes</td></tr>
<tr><td>B</td><td>A second device and an accomplice</td><td>Video call with an accomplice who performs the prompts, shown on a screen in front of the camera</td><td>Partly, see below</td></tr>
<tr><td>C</td><td>Specialist materials</td><td>Silicone or resin mask, 3D printed head</td><td>No</td></tr>
<tr><td>C</td><td>Control of the capture device</td><td>Virtual camera, injected video stream, rooted phone</td><td>No</td></tr>
<tr><td>C</td><td>Real time synthesis</td><td>Deepfake puppet that follows prompts, injected or shown on a screen</td><td>Partly, see below</td></tr>
</table>

## Which signal stops which attack

<table>
<tr><th>Attack</th><th>Passive texture model</th><th>Random challenges</th><th>Depth from motion</th><th>Remote pulse</th></tr>
<tr><td>Printed photograph held still</td><td>Detects paper texture, halftone, edges</td><td>Fails every prompt</td><td>Not triggered</td><td>No pulse</td></tr>
<tr><td>Printed photograph tilted to fake a head turn</td><td>Detects</td><td>Mostly fails, the estimated yaw barely moves</td><td>Detects a flat surface</td><td>No pulse</td></tr>
<tr><td>Photograph on a phone</td><td>Detects bezel, moire, emission</td><td>Fails every prompt</td><td>Detects a flat surface if tilted</td><td>No pulse</td></tr>
<tr><td>Recorded video replayed on a screen</td><td>Detects</td><td>Fails, prompts and timing are random</td><td>Passes, the recording shows real 3D motion</td><td>Usually no pulse</td></tr>
<tr><td>Paper mask with eye holes</td><td>Detects paper and cut edge</td><td>Blink can pass, turns are tracked on the paper</td><td>Detects a flat surface</td><td>No pulse on paper</td></tr>
<tr><td>Live accomplice relayed on a screen</td><td>The only signal that sees the screen</td><td>Passes</td><td>Passes</td><td>May pass</td></tr>
<tr><td>Deepfake shown on a screen</td><td>Sees the screen, not the synthesis</td><td>Depends on the puppet</td><td>Depends on the puppet</td><td>Usually no pulse</td></tr>
</table>

The table shows why the signals are combined with hard gates instead of a single averaged score. No signal covers every row, and every row is covered by at least one signal that an attacker of level A or B cannot easily defeat.

## Design decisions that follow from the threat model

**Prompts are random in content, order and timing.** The pool holds five actions and a session draws three without repeats, which gives 60 ordered sequences before timing is considered. The pause before each prompt is random as well. A response that arrives faster than human reaction time is ignored, so a recording that happens to be moving when the prompt appears gains nothing.

**A wrong turn fails immediately.** When the prompt says left and the head goes right, the challenge fails without waiting for the timeout. A recording that contains both turns therefore cannot simply play through.

**The face must stay the same face.** During the interactive stages the session ends if the face disappears for more than a second, jumps across the frame between two frames, changes size abruptly or is joined by a second face. This closes the gap between passing the challenges and being scored.

**Server time is authoritative.** The HTTP service timestamps frames on arrival and ignores client clocks, because response timing is part of the evidence.

**The passive model is a gate, not only a vote.** A relayed accomplice passes everything except the texture model, so a passive score far below its threshold rejects the session whatever else happened.

**Pulse can support a decision but never make one.** Remote pulse measurement is fragile under movement, compression and poor light. It adds confidence when it is clearly present and costs a real user very little when it is not.

**Inconclusive is a separate outcome.** Poor light, a face too far away or a session that ran out of time are not evidence of an attack. They are reported as inconclusive with the reason, so the calling application can ask the user to try again instead of locking them out.

## Out of scope and why

**Injection attacks.** If the attacker controls the video stream that reaches the software, no analysis of pixels alone is sufficient. Defences live elsewhere: device attestation, signed capture components, secure camera pipelines.

**High quality 3D masks.** Detecting silicone masks reliably needs sensors this project does not assume, such as depth or near infrared cameras.

**Deepfakes as such.** PulseGate AI detects the screen a deepfake is shown on. It does not analyse synthesis artefacts, and it has no defence against a deepfake injected directly into the stream.

**Morphing and document fraud.** These belong to the identity document and face matching steps.

## Residual risks an integrator should know

* The passive model was trained on simulated attacks. Its error rates on real attack instruments are not measured in this repository. The section on limitations in the README and the model card explain what was and was not validated, and how to calibrate on real captures.
* Thresholds trade convenience against security. The shipped operating point is chosen on validation data from the simulator. A deployment should set it from genuine and attack captures of its own cameras and users.
* Repeated attempts multiply an attacker's chances. The calling application should limit retries per account and per device.
* Results are logged per session as JSON without images. Whether images are kept is a decision for the integrator and has data protection consequences.
