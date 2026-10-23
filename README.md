# RVC Studio

RVC Studio is a local, real-time voice conversion desktop application built with Avalonia and ONNX Runtime. It captures microphone audio, runs it through an RVC model, and sends the converted signal to a selected output device. Audio and voice models stay on the user's machine.

> This project is under active development. You need a compatible RVC model and the ContentVec/RMVPE base models. RVC Studio does not ship user voice models.

## Features

- Real-time microphone-to-output conversion through PortAudio.
- Import ZIP archives containing an RVC ONNX generator or `.pth` checkpoint.
- Automatic `.pth` to ONNX conversion using the bundled PyInstaller converter.
- Optional retrieval-index preparation and runtime blending.
- Pitch shift, input/output device selection, and live audio diagnostics.
- CUDA-first inference on supported Windows/Linux systems with automatic CPU fallback.

## Backend overview

### Desktop application

The UI is an Avalonia desktop app targeting .NET 10. `MainViewModel` discovers local models and PortAudio devices, handles model archive import, owns the real-time engine lifecycle, and exposes conversion controls. Imported models are stored in `Models/` beside the application. Archive extraction checks resolved paths to prevent ZIP path traversal.

When importing an archive, the app first looks for a ready ONNX generator. If it finds one, it normalizes its location to `model.onnx`; if a `.index` file is also present, it invokes the converter to build the compact `features.rvci` index representation. If the archive only contains a `.pth` checkpoint, the app launches the platform-specific `rvc-converter` executable and verifies that conversion produced an ONNX file.

### Real-time audio pipeline

`RealtimeAudioEngine` uses PortAudio for full-duplex, mono audio at 48 kHz. Its audio callback is deliberately lightweight: it copies samples into preallocated buffers, applies a simple noise gate, and transfers complete windows through bounded queues. ONNX inference runs on a dedicated worker thread rather than on the device callback. Small buffer pools limit allocations and reduce the risk of audio glitches. Diagnostic counters expose RMS levels, inference time, and output underruns.

`RvcOnnxPipeline` processes each input window in stages:

1. Downsample 48 kHz audio to 16 kHz for feature extraction.
2. Run ContentVec to obtain frame-level speech features.
3. Optionally run RMVPE to estimate F0, then apply pitch shift and quantization.
4. Optionally blend features with the imported model's retrieval index.
5. Feed features, pitch, speaker ID, and noise inputs to the RVC ONNX generator.
6. Resample generated audio to the output stream rate and crossfade block boundaries.

Each ONNX session tries the CUDA execution provider first on Windows/Linux. If the CUDA provider or its native libraries cannot initialize, that session is created with the CPU provider. macOS currently uses CPU execution. NVIDIA acceleration requires a compatible NVIDIA driver and CUDA/cuDNN runtime libraries. The NuGet package includes ONNX Runtime GPU native providers, but it does not install system GPU drivers or CUDA dependencies.

### Python converter

`RVCStudio/converter/convert.py` is the offline model-conversion backend. It exports RVC checkpoints to ONNX and converts FAISS retrieval indexes into a compact format consumed by the C# audio pipeline. `rvc_exporter/` contains the model/export helpers. PyInstaller packages this backend as a platform-native executable so end users do not need Python installed.

The root setup scripts create an isolated build virtual environment, install `requirements.txt`, invoke PyInstaller, download and verify the shared ONNX base models, restore .NET packages, and build the application. PyInstaller output is platform-specific, so run setup on the same OS and architecture you intend to use.

### Models and runtime files

- `RVCStudio/assets/models/contentvec.onnx` — ContentVec feature extractor.
- `RVCStudio/assets/models/rmvpe.onnx` — RMVPE pitch estimator.
- `Models/<voice>/model.onnx` — imported user's RVC generator.
- `Models/<voice>/features.rvci` — optional prepared retrieval features.

Base model files are downloaded by setup and intentionally excluded from Git due to their size. Their license and notices are kept under `RVCStudio/assets/models/`.

## Installation

### Windows

1. Install the [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0), Python 3.10–3.12, and Git. Ensure `python` and `curl.exe` are available in PowerShell.
2. Install a PortAudio runtime if your environment does not already provide it. For NVIDIA GPU inference, install a compatible NVIDIA driver and CUDA/cuDNN runtime.
3. From the repository root, run:

   ```powershell
   .\setup.ps1
   ```

4. Start the app:

   ```powershell
   dotnet run --project RVCStudio/RVCStudio.csproj
   ```

### Linux

1. Install the [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0), Python 3 with `venv`, PortAudio, `curl`, and `sha256sum`. On Debian/Ubuntu:

   ```bash
   sudo apt install python3 python3-venv libportaudio2 curl coreutils
   ```

2. For NVIDIA inference, install a compatible NVIDIA driver and CUDA/cuDNN runtime.
3. From the repository root, run:

   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

4. Start the app:

   ```bash
   dotnet run --project RVCStudio/RVCStudio.csproj
   ```

Setup downloads approximately 707 MiB of base models and verifies their SHA-256 hashes. PyTorch/PyInstaller dependencies can require additional disk space and network bandwidth.

## Manual build and publish

After running setup, build the solution with `dotnet build RVCStudio.sln -c Release`. To publish a self-contained Linux x64 directory:

```bash
dotnet publish RVCStudio/RVCStudio.csproj -c Release -r linux-x64 --self-contained true -o publish/linux-x64
```

Use `win-x64` (or another supported RID) for Windows. Build the converter on the target platform before publishing; the setup script does this automatically.

## License and model terms

Review the licenses of dependencies and model files before redistribution. The base model provenance and license notices are in `RVCStudio/assets/models/THIRD_PARTY_NOTICES.md`. User-provided RVC checkpoints and indexes may have their own usage and redistribution terms.
