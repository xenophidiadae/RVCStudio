# RVC Studio

**RVC Studio is a desktop voice changer that converts your microphone input into another voice in real time using RVC (Retrieval-based Voice Conversion) models.** Choose a voice model, select an input and output device, adjust the pitch shift, and start the voice changer. Audio is processed locally on your computer; the app does not train models or provide voice models.

> **Beta:** RVC Studio is still in beta

## How it works

RVC Studio captures audio from the selected microphone, analyzes speech, runs it through the selected RVC voice model, and plays the converted audio through the selected output device. You can route that output to headphones, speakers, or a virtual audio cable for use in another app.

The app is built with C# and .NET 10. Avalonia provides the desktop UI, PortAudio handles audio capture and playback, and ONNX Runtime runs the neural networks. The conversion pipeline uses ContentVec to extract speech features, an RVC generator to synthesize the target voice, and RMVPE for pitch detection when required by the selected model. Inference runs on the CPU by default; CUDA acceleration is available on supported Windows and Linux systems with a compatible CUDA environment.

Audio capture and model inference run on separate threads, connected by short queues of audio blocks. This keeps model processing out of the real-time audio callback and helps limit latency. The app also smooths audio block boundaries to reduce clicks and gaps.

## Adding a voice model

Drag and drop a model ZIP into the app or use the file picker. The archive can contain either a compatible RVC generator in ONNX format or an RVC v2 `.pth` checkpoint. ONNX models are imported directly; `.pth` checkpoints are converted to ONNX by the bundled converter. Imported models are stored in the `Models` folder next to the app.

RVC Studio does not include a target voice model. You must provide one yourself and follow its license and usage terms. The ContentVec and RMVPE support models are downloaded during setup into `RVCStudio/assets/models/` (about 707 MiB total).

## Project layout

- `RVCStudio/` — Avalonia desktop app, user interface, audio engine, and ONNX inference pipeline.
- `RVCStudio/converter/` — source and build scripts for converting RVC v2 `.pth` checkpoints to ONNX.
- `RVCStudio/assets/models/` — downloaded ContentVec/RMVPE support models and third-party notices.
- `scripts/` — setup and run scripts for Windows and Linux.

## Installation

Install the **.NET 10 SDK** first. On Linux, also install PortAudio and `curl` (Debian/Ubuntu: `sudo apt install libportaudio2 curl`). Run setup from the repository root; it downloads the support models and restores the .NET dependencies.

**Windows (PowerShell):**

```powershell
.\scripts\setup.ps1
.\scripts\run.ps1
```

**Linux:**

```bash
./scripts/setup.sh
./scripts/run.sh
```

After setup, use the `run` script to start the app again. For a Release build, run `dotnet build RVCStudio.sln -c Release`.

## Licensing

=======
Check the licenses of dependencies and models before redistributing the app. Third-party model notices are in `RVCStudio/assets/models/THIRD_PARTY_NOTICES.md`; user-provided voice models may have separate terms.
=======
