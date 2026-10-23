using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Avalonia.Platform.Storage;
using PortAudioSharp;
using System.Collections.ObjectModel;
using System.IO.Compression;
using System.Threading.Tasks;
using System;
using System.IO;
using System.Linq;
using System.Diagnostics;
using System.Threading;
using RVCStudio.Audio;

namespace RVCStudio.ViewModels;

public partial class MainViewModel : ViewModelBase, IDisposable
{
    public ObservableCollection<string> Models { get; } = [];
    public ObservableCollection<AudioDeviceInfo> InputDevices { get; } = [];
    public ObservableCollection<AudioDeviceInfo> OutputDevices { get; } = [];

    [ObservableProperty] public partial string? SelectedModel { get; set; }
    [ObservableProperty] public partial AudioDeviceInfo? SelectedInputDevice { get; set; }
    [ObservableProperty] public partial AudioDeviceInfo? SelectedOutputDevice { get; set; }
    [ObservableProperty] public partial string StatusMessage { get; set; } = "";
    [ObservableProperty] public partial string AudioStatus { get; set; } = "";
    [ObservableProperty] public partial int PitchShift { get; set; }
    [ObservableProperty] public partial double IndexRate { get; set; } = 0.75;
    [ObservableProperty] public partial bool IsVoiceChanging { get; set; }
    public string VoiceChangerButtonText => IsVoiceChanging ? "Stop voice conversion" : "Start voice conversion";

    partial void OnIsVoiceChangingChanged(bool value) => OnPropertyChanged(nameof(VoiceChangerButtonText));

    private readonly string _modelsDirectory = Path.Combine(AppContext.BaseDirectory, "Models");
    private readonly string _applicationDirectory = AppContext.BaseDirectory;
    private RealtimeAudioEngine? _audioEngine;
    private Timer? _audioDiagnosticsTimer;
    private long _lastReportedUnderrunCallbacks;
    private long _lastReportedUnderrunFrames;

    public MainViewModel()
    {
        Directory.CreateDirectory(_modelsDirectory);
        PortAudioLifetime.EnsureInitialized();
        RefreshModels();
        RefreshAudioDevices();
    }

    partial void OnPitchShiftChanged(int value)
    {
        if (_audioEngine is not null) _audioEngine.PitchShiftSemitones = value;
    }

    partial void OnIndexRateChanged(double value)
    {
        if (_audioEngine is not null) _audioEngine.IndexRate = (float)value;
    }

    [RelayCommand]
    private async Task BrowseModel()
    {
        var window = Avalonia.Application.Current?.ApplicationLifetime is Avalonia.Controls.ApplicationLifetimes.IClassicDesktopStyleApplicationLifetime desktop
            ? desktop.MainWindow : null;
        if (window is null) return;
        var files = await window.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Select an RVC model archive",
            AllowMultiple = false,
            FileTypeFilter = [new FilePickerFileType("ZIP archive") { Patterns = ["*.zip"] }]
        });
        if (files.Count > 0) await ImportArchive(files[0].TryGetLocalPath());
    }

    public async Task ImportArchive(string? archivePath)
    {
        if (string.IsNullOrWhiteSpace(archivePath) || !File.Exists(archivePath) || !archivePath.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
        {
            StatusMessage = "Select an existing ZIP archive.";
            return;
        }

        string? importDirectory = null;
        try
        {
            var name = Path.GetFileNameWithoutExtension(archivePath);
            var destination = Path.Combine(_modelsDirectory, name);
            var suffix = 2;
            while (Directory.Exists(destination)) destination = Path.Combine(_modelsDirectory, $"{name} ({suffix++})");
            Directory.CreateDirectory(destination);
            importDirectory = destination;
            var root = Path.GetFullPath(destination) + Path.DirectorySeparatorChar;
            using var zip = ZipFile.OpenRead(archivePath);
            string? onnxInArchive = null;
            string? pthInArchive = null;
            foreach (var entry in zip.Entries)
            {
                var target = Path.GetFullPath(Path.Combine(destination, entry.FullName));
                if (!target.StartsWith(root, StringComparison.Ordinal)) throw new InvalidDataException("The archive contains an unsafe path.");
                if (string.IsNullOrEmpty(entry.Name)) Directory.CreateDirectory(target);
                else
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(target)!);
                    entry.ExtractToFile(target, overwrite: false);
                    if (entry.FullName.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) &&
                        !Path.GetFileName(entry.FullName).Equals("contentvec.onnx", StringComparison.OrdinalIgnoreCase) &&
                        !Path.GetFileName(entry.FullName).Equals("rmvpe.onnx", StringComparison.OrdinalIgnoreCase))
                        onnxInArchive ??= target;
                    if (entry.FullName.EndsWith(".pth", StringComparison.OrdinalIgnoreCase))
                        pthInArchive ??= target;
                }
            }

            if (onnxInArchive is not null)
            {
                var targetGenerator = Path.Combine(destination, "model.onnx");
                if (!string.Equals(onnxInArchive, targetGenerator, StringComparison.Ordinal))
                    File.Copy(onnxInArchive, targetGenerator, overwrite: true);
                var indexPath = Directory.EnumerateFiles(destination, "*.index", SearchOption.AllDirectories).FirstOrDefault();
                    if (indexPath is not null)
                {
                    var indexConverterPath = GetConverterPath();
                    if (!File.Exists(indexConverterPath)) throw new FileNotFoundException("The converter required to prepare the RVC index was not found.", indexConverterPath);
                    await RunIndexPreparation(indexConverterPath, indexPath, Path.Combine(destination, "features.rvci"));
                }
                RefreshModels();
                SelectedModel = Path.GetFileName(destination);
                StatusMessage = $"ONNX model '{SelectedModel}' imported. Python conversion was not needed.";
                return;
            }
            if (pthInArchive is null)
                throw new InvalidDataException("The ZIP contains neither an RVC .pth checkpoint nor an ONNX generator.");

            var converterPath = GetConverterPath();
            if (!File.Exists(converterPath))
                throw new FileNotFoundException(
                    $"The RVC converter for this platform was not found ({Path.GetFileName(converterPath)}). " +
                    "Run the setup script for this operating system.", converterPath);

            StatusMessage = "Converting .pth to ONNX… This may take a few minutes.";
            var startInfo = new ProcessStartInfo
            {
                FileName = converterPath,
                WorkingDirectory = Path.GetDirectoryName(converterPath)!,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };
            startInfo.ArgumentList.Add(archivePath);
            startInfo.ArgumentList.Add("--output");
            startInfo.ArgumentList.Add(destination);

            using var process = Process.Start(startInfo)
                ?? throw new InvalidOperationException("Could not start the RVC converter.");
            var stdoutTask = process.StandardOutput.ReadToEndAsync();
            var stderrTask = process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();
            var output = await stdoutTask;
            var error = await stderrTask;
            if (process.ExitCode != 0)
            {
                var details = string.IsNullOrWhiteSpace(error) ? output : error;
                throw new InvalidOperationException(string.IsNullOrWhiteSpace(details)
                    ? $"The converter exited with code {process.ExitCode}."
                    : $"The converter could not process the model: {details.Trim()}");
            }
            if (!File.Exists(Path.Combine(destination, $"{name}.onnx")))
                throw new InvalidOperationException("The converter exited without creating an ONNX file.");

            RefreshModels();
            SelectedModel = Path.GetFileName(destination);
            StatusMessage = $"Model '{SelectedModel}' was imported and converted to ONNX.";
        }
        catch (Exception ex)
        {
            try
            {
                if (importDirectory is not null && Directory.Exists(importDirectory))
                    Directory.Delete(importDirectory, recursive: true);
            }
            catch (IOException cleanupError)
            {
                StatusMessage = $"Could not clean up the incomplete model: {cleanupError.Message}";
            }
            StatusMessage = $"Could not import archive: {ex.Message}";
        }
    }

    private string GetConverterPath()
    {
        var platformDirectory = OperatingSystem.IsWindows() ? "windows-x64" : "linux-x64";
        var fileName = OperatingSystem.IsWindows() ? "rvc-converter.exe" : "rvc-converter";
        var platformPath = Path.Combine(_applicationDirectory, "converter", "dist", platformDirectory, fileName);
        return File.Exists(platformPath) ? platformPath : Path.Combine(_applicationDirectory, "converter", "dist", fileName);
    }

    private static async Task RunIndexPreparation(string converterPath, string indexPath, string outputPath)
    {
        var info = new ProcessStartInfo
        {
            FileName = converterPath, WorkingDirectory = Path.GetDirectoryName(converterPath)!,
            UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true, CreateNoWindow = true
        };
        info.ArgumentList.Add("--prepare-index");
        info.ArgumentList.Add(indexPath);
        info.ArgumentList.Add(outputPath);
        using var process = Process.Start(info) ?? throw new InvalidOperationException("Could not start RVC index preparation.");
        var stdout = process.StandardOutput.ReadToEndAsync();
        var stderr = process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();
        var output = await stdout;
        var error = await stderr;
        if (process.ExitCode != 0)
            throw new InvalidOperationException($"Could not prepare the RVC index: {(string.IsNullOrWhiteSpace(error) ? output : error).Trim()}");
    }

    private void RefreshModels()
    {
        Models.Clear();
        foreach (var directory in Directory.EnumerateDirectories(_modelsDirectory)) Models.Add(Path.GetFileName(directory));
        foreach (var archive in Directory.EnumerateFiles(_modelsDirectory, "*.zip")) Models.Add(Path.GetFileNameWithoutExtension(archive));
        SelectedModel ??= Models.FirstOrDefault();
    }

    private void RefreshAudioDevices()
    {
        try
        {
            InputDevices.Clear();
            OutputDevices.Clear();
            for (var i = 0; i < PortAudio.DeviceCount; i++)
            {
                var device = PortAudio.GetDeviceInfo(i);
                if (device.maxInputChannels > 0) InputDevices.Add(new AudioDeviceInfo(i, device.name));
                if (device.maxOutputChannels > 0) OutputDevices.Add(new AudioDeviceInfo(i, device.name));
            }
            SelectedInputDevice = InputDevices.FirstOrDefault();
            SelectedOutputDevice = OutputDevices.FirstOrDefault();
            AudioStatus = InputDevices.Count == 0 || OutputDevices.Count == 0
                ? "Some devices were not found. Connect them and restart the app."
                : "Devices loaded from the system. Restart the app after connecting another device.";
        }
        catch (Exception ex) { AudioStatus = $"Could not enumerate audio devices: {ex.Message}"; }
    }

    [RelayCommand]
    private void ToggleVoiceChanging()
    {
        if (IsVoiceChanging)
        {
            StopVoiceChanging();
            return;
        }

        try
        {
            if (string.IsNullOrWhiteSpace(SelectedModel))
                throw new InvalidOperationException("Import and select a model first.");
            var modelDir = Path.Combine(_modelsDirectory, SelectedModel);
            if (SelectedInputDevice is not { } inputDevice)
                throw new InvalidOperationException("Select an input device.");
            if (SelectedOutputDevice is not { } outputDevice)
                throw new InvalidOperationException("Select an output device.");
            var systemModelsDir = Path.Combine(_applicationDirectory, "assets", "models");
            _audioEngine = new RealtimeAudioEngine(modelDir, systemModelsDir, inputDevice.Id, outputDevice.Id)
            {
                PitchShiftSemitones = PitchShift,
                IndexRate = (float)IndexRate
            };
            _audioEngine.Start();
            _lastReportedUnderrunCallbacks = 0;
            _lastReportedUnderrunFrames = 0;
            IsVoiceChanging = true;
            AudioStatus = $"Starting realtime · input #{inputDevice.Id}: {inputDevice.Name} · " +
                          $"output #{outputDevice.Id}: {outputDevice.Name} · collecting metrics…";
            _audioDiagnosticsTimer?.Dispose();
            _audioDiagnosticsTimer = new Timer(_ => UpdateAudioDiagnostics(), null, 1000, 1000);
        }
        catch (Exception ex)
        {
            _audioEngine?.Dispose();
            _audioEngine = null;
            IsVoiceChanging = false;
            AudioStatus = $"Could not start realtime conversion: {ex.Message}";
        }
    }

    private void StopVoiceChanging()
    {
        _audioDiagnosticsTimer?.Dispose();
        _audioDiagnosticsTimer = null;
        try { _audioEngine?.Dispose(); }
        catch (Exception ex) { AudioStatus = $"Audio shutdown error: {ex.Message}"; }
        _audioEngine = null;
        IsVoiceChanging = false;
        if (string.IsNullOrEmpty(AudioStatus)) AudioStatus = "Realtime voice conversion stopped.";
    }

    public void Dispose()
    {
        StopVoiceChanging();
    }

    private void UpdateAudioDiagnostics()
    {
        var engine = _audioEngine;
        if (engine is null || !IsVoiceChanging) return;
        Avalonia.Threading.Dispatcher.UIThread.Post(() =>
        {
            if (!ReferenceEquals(engine, _audioEngine)) return;
            AudioStatus = engine.WorkerError is { } workerError
                ? $"ONNX worker error: {workerError.Message}"
                : $"Realtime · input RMS {engine.LastInputRms:F4} · output RMS {engine.LastOutputRms:F4} · " +
                  $"inferences {engine.InferenceBlocks} · last {engine.LastInferenceMilliseconds} ms · " +
                  $"underruns/s {engine.OutputUnderrunCallbacks - _lastReportedUnderrunCallbacks} callbacks, " +
                  $"{engine.OutputUnderrunFrames - _lastReportedUnderrunFrames} silent frames";
            _lastReportedUnderrunCallbacks = engine.OutputUnderrunCallbacks;
            _lastReportedUnderrunFrames = engine.OutputUnderrunFrames;
        });
    }
}
