using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace RVCStudio.Audio;
public sealed class RvcOnnxPipeline : IDisposable
{
    private readonly InferenceSession _content;
    private readonly InferenceSession? _pitch;
    private readonly InferenceSession _generator;
    private readonly string _contentInput;
    private readonly string? _pitchInput;
    private readonly string _generatorPhone;
    private readonly string? _generatorLengths;
    private readonly string? _generatorPitchLengths;
    private readonly string? _generatorPitch;
    private readonly string? _generatorPitchF;
    private readonly string? _generatorSpeaker;
    private readonly string? _generatorNoise;
    private readonly string? _generatorRnd;
    private readonly string? _generatorSid;
    private readonly string _generatorOutput;
    private readonly int _contentFrameShift;
    private readonly int _outputSampleRate;
    private const int InputWindow16k = 4800;
    private readonly float[] _audio16k = new float[InputWindow16k];
    private readonly long[] _quantizedPitch = new long[512];
    private readonly float[] _shiftedF0 = new float[512];
    private readonly float[] _noise = new float[192 * 512];
    private readonly long[] _zeroPitch = new long[512];
    private readonly float[] _features = new float[512 * 768];
    private readonly float[] _rawContent = new float[512 * 768];
    private readonly float[] _rawPitch = new float[512];
    private readonly float[] _outputAudio = new float[24000];
    private readonly long[] _speaker = [0];
    private const int BoundaryCrossfadeSamples = 240;
    private float _previousLastSample;
    private readonly float[] _currentOutput = new float[12000];
    private readonly int[] _audioInputShape = [1, InputWindow16k];
    private readonly int[] _phoneShape = [1, 512, 768];
    private readonly int[] _pitchShape = [1, 512];
    private readonly int[] _noiseShape = [1, 192, 512];
    private readonly int[] _speakerShape = [1];
    private readonly long[] _phoneLength = [512];
    private readonly float[] _pitchThreshold = [0.03f];
    private readonly int[] _thresholdShape = [1];
    private readonly List<NamedOnnxValue> _contentInputs = new(1);
    private readonly List<NamedOnnxValue> _pitchInputs = new(1);
    private readonly List<NamedOnnxValue> _generatorInputs = new(7);
    private readonly Random _noiseRandom = new(17391);
    private readonly RvcFeatureIndex? _featureIndex;
    private bool _hasTail;

    public RvcOnnxPipeline(string modelDirectory, string systemModelsDirectory)
    {
        var contentPath = Path.Combine(systemModelsDirectory, "contentvec.onnx");
        var pitchPath = Path.Combine(systemModelsDirectory, "rmvpe.onnx");
        var generatorPath = Path.Combine(modelDirectory, "model.onnx");
        if (!File.Exists(generatorPath))
            generatorPath = Directory.EnumerateFiles(modelDirectory, "*.onnx", SearchOption.AllDirectories)
                .FirstOrDefault(path => !string.Equals(Path.GetFileName(path), "contentvec.onnx", StringComparison.OrdinalIgnoreCase)
                    && !string.Equals(Path.GetFileName(path), "rmvpe.onnx", StringComparison.OrdinalIgnoreCase)) ?? generatorPath;
        if (!File.Exists(contentPath) || !File.Exists(generatorPath))
            throw new FileNotFoundException(
                $"Required models were not found: '{contentPath}' and/or an ONNX generator in '{modelDirectory}'. " +
                "Place ContentVec in assets/models and the target model in the selected model folder.");

        _content = CreateSession(contentPath);
        _generator = CreateSession(generatorPath);
        _contentInput = FindOptionalInput(_content, "audio") ?? _content.InputMetadata.Keys.First();
        _generatorPhone = FindInput(_generator, "phone", "feats", "features");
        _generatorLengths = FindOptionalInput(_generator, "phone_lengths");
        _generatorPitchLengths = FindOptionalInput(_generator, "p_len");
        _generatorPitch = FindOptionalInput(_generator, "pitch");
        _generatorPitchF = FindOptionalInput(_generator, "pitchf");
        _generatorSpeaker = FindOptionalInput(_generator, "ds", "speaker_id");
        _generatorSid = FindOptionalInput(_generator, "sid");
        _generatorNoise = FindOptionalInput(_generator, "rnd", "noise");
        _generatorRnd = FindOptionalInput(_generator, "rnd");
        UsesPitch = _generatorPitchF is not null || _generatorPitch is not null;
        if (UsesPitch)
        {
            if (!File.Exists(pitchPath)) throw new FileNotFoundException($"The generator requires pitch input, but the RMVPE model was not found: {pitchPath}");
            _pitch = CreateSession(pitchPath);
            _pitchInput = FindOptionalInput(_pitch, "waveform") ?? _pitch.InputMetadata.Keys.First();
        }
        _generatorOutput = _generator.OutputMetadata.Keys.First();

        if (_generatorSpeaker is null)
            throw new InvalidDataException("The ONNX generator has no speaker/ds input.");
        var contentOutputName = FindOptionalOutput(_content, "unit12");
        var contentShape = _content.OutputMetadata[contentOutputName ?? _content.OutputMetadata.Keys.First()].Dimensions;
        _contentFrameShift = contentOutputName is not null
            ? 2
            : contentShape.Length >= 3 && contentShape[1] > 0 && contentShape[1] <= 64 ? 2 : 1;
        _outputSampleRate = ResolveOutputSampleRate(modelDirectory, _generator);
        var preparedIndexPath = Path.Combine(modelDirectory, "features.rvci");
        if (File.Exists(preparedIndexPath)) _featureIndex = RvcFeatureIndex.Load(preparedIndexPath);
    }

    public int OutputSampleRate => _outputSampleRate;
    public bool UsesPitch { get; }

    public ReadOnlySpan<float> Convert(ReadOnlySpan<float> input48k, int pitchShiftSemitones, float indexRate)
    {
        ResampleTo16k(input48k, _audio16k);
        _contentInputs.Clear();
        _contentInputs.Add(NamedOnnxValue.CreateFromTensor(_contentInput,
            new DenseTensor<float>(_audio16k, _audioInputShape)));
        using var contentResults = _content.Run(_contentInputs);
        var featuresValue = contentResults.FirstOrDefault(x => x.Name == "unit12") ?? contentResults.First();
        var featuresTensor = featuresValue.AsTensor<float>();
        if (featuresTensor.Dimensions.Length != 3) throw new InvalidDataException("ContentVec must return [1,T,C].");
        var baseFrames = featuresTensor.Dimensions[1];
        var inputFeatures = featuresTensor.Dimensions[2];
        if (inputFeatures > 768) throw new InvalidDataException("ContentVec dimensions above 768 are not supported.");
        var generatorFeatureDims = _generator.InputMetadata[_generatorPhone].Dimensions;
        var generatorFeatureSize = generatorFeatureDims.Length >= 3 && generatorFeatureDims[^1] > 0
            ? generatorFeatureDims[^1] : inputFeatures;
        var frames = Math.Min(baseFrames * _contentFrameShift, _shiftedF0.Length);
        if (frames < 1) throw new InvalidDataException("ContentVec returned an empty feature tensor.");
        if (inputFeatures != generatorFeatureSize)
            throw new InvalidDataException($"ContentVec dimensions ({inputFeatures}) do not match the generator input ({generatorFeatureSize}). A matching HuBERT model is required.");
        if (featuresTensor is not DenseTensor<float> denseFeatures)
            throw new InvalidDataException("ContentVec returned a tensor without a contiguous DenseTensor buffer.");
        denseFeatures.Buffer.Span.CopyTo(_rawContent);
        Array.Clear(_features);
        _phoneShape[1] = frames;
        _phoneShape[2] = generatorFeatureSize;
        for (var t = 0; t < frames; t++)
            Array.Copy(_rawContent, (t / _contentFrameShift) * inputFeatures,
                _features, t * generatorFeatureSize, generatorFeatureSize);
        _featureIndex?.Blend(_features, frames, generatorFeatureSize, indexRate);
        var frameCount = frames;
        if (UsesPitch)
        {
            _pitchInputs.Clear();
            _pitchInputs.Add(NamedOnnxValue.CreateFromTensor(_pitchInput!,
                new DenseTensor<float>(_audio16k, _audioInputShape)));
            if (_pitch!.InputMetadata.ContainsKey("threshold"))
                _pitchInputs.Add(NamedOnnxValue.CreateFromTensor("threshold",
                    new DenseTensor<float>(_pitchThreshold, _thresholdShape)));
            using var pitchResults = _pitch!.Run(_pitchInputs);
            var f0Value = pitchResults.FirstOrDefault(x => x.Name == "pitchf") ?? pitchResults.First();
            var f0Tensor = f0Value.AsTensor<float>();
            if (f0Tensor.Length > _rawPitch.Length) throw new InvalidDataException("RMVPE frame count exceeds the supported window size.");
            if (f0Tensor is not DenseTensor<float> denseF0)
                throw new InvalidDataException("RMVPE returned a tensor without a contiguous DenseTensor buffer.");
            denseF0.Buffer.Span.CopyTo(_rawPitch);
            frameCount = (int)Math.Min(frames, f0Tensor.Length);
            if (frameCount == 0) throw new InvalidDataException("RMVPE returned empty F0 data.");
        }
        var alignedFrames = frameCount;
        _generatorInputs.Clear();
        _phoneShape[1] = alignedFrames;
        _phoneLength[0] = alignedFrames;
        var phoneTensor = new DenseTensor<float>(new Memory<float>(_features, 0, alignedFrames * generatorFeatureSize), _phoneShape);
        _generatorInputs.Add(NamedOnnxValue.CreateFromTensor(_generatorPhone, phoneTensor));
        if (_generatorLengths is not null || _generatorPitchLengths is not null)
        {
            _phoneLength[0] = alignedFrames;
            _generatorInputs.Add(NamedOnnxValue.CreateFromTensor(_generatorLengths ?? _generatorPitchLengths!,
                new DenseTensor<long>(_phoneLength, _speakerShape)));
        }

        var ratio = (float)Math.Pow(2.0, pitchShiftSemitones / 12.0);
        const float f0Min = 50f;
        const float f0Max = 1100f;
        var melMin = 1127f * MathF.Log(1f + f0Min / 700f);
        var melMax = 1127f * MathF.Log(1f + f0Max / 700f);
        for (var i = 0; UsesPitch && i < frameCount; i++)
        {
            var hz = _rawPitch[i] > 0 ? _rawPitch[i] * ratio : 0;
            _shiftedF0[i] = hz;
            if (hz <= 0)
            {
                _quantizedPitch[i] = 1;
            }
            else
            {
                var mel = 1127f * MathF.Log(1f + hz / 700f);
                var coarse = (mel - melMin) * 254f / (melMax - melMin) + 1f;
                _quantizedPitch[i] = (long)Math.Clamp(MathF.Round(coarse), 1f, 255f);
            }
        }
        if (_generatorPitch is not null || _generatorPitchF is not null)
        {
            _pitchShape[1] = frameCount;
            if (_generatorPitch is not null)
                _generatorInputs.Add(NamedOnnxValue.CreateFromTensor(_generatorPitch,
                    new DenseTensor<long>(new Memory<long>(_quantizedPitch, 0, frameCount), _pitchShape)));
            if (_generatorPitchF is not null)
                _generatorInputs.Add(NamedOnnxValue.CreateFromTensor(_generatorPitchF,
                    new DenseTensor<float>(new Memory<float>(_shiftedF0, 0, frameCount), _pitchShape)));
        }
        else
        {
            _pitchShape[1] = alignedFrames;
            if (_generatorInputs.All(x => x.Name != "pitch"))
                _generatorInputs.Add(NamedOnnxValue.CreateFromTensor("pitch",
                    new DenseTensor<long>(new Memory<long>(_zeroPitch, 0, alignedFrames), _pitchShape)));
            if (_generatorInputs.All(x => x.Name != "pitchf"))
                _generatorInputs.Add(NamedOnnxValue.CreateFromTensor("pitchf",
                    new DenseTensor<float>(new Memory<float>(_shiftedF0, 0, alignedFrames), _pitchShape)));
        }
        if (_generatorSpeaker is not null || _generatorSid is not null)
            _generatorInputs.Add(NamedOnnxValue.CreateFromTensor(_generatorSpeaker ?? _generatorSid!,
                new DenseTensor<long>(_speaker, _speakerShape)));
        if (_generatorNoise is not null || _generatorRnd is not null)
        {
            _noiseShape[2] = frameCount;
            FillGaussianNoise(_noise, 192 * frameCount);
            _generatorInputs.Add(NamedOnnxValue.CreateFromTensor(_generatorNoise ?? _generatorRnd!,
                new DenseTensor<float>(new Memory<float>(_noise, 0, 192 * frameCount), _noiseShape)));
        }

        using var generated = _generator.Run(_generatorInputs);
        var audio = generated.First(x => x.Name == _generatorOutput).AsTensor<float>();
        var audioArray = _outputAudio;
        if (audio.Length > audioArray.Length) throw new InvalidDataException("The generator returned an oversized audio tensor.");
        if (audio is not DenseTensor<float> denseAudio)
            throw new InvalidDataException("Generator returned a tensor without a contiguous DenseTensor buffer.");
        denseAudio.Buffer.Span.CopyTo(audioArray);
        var usable = Math.Min(audio.Length / 2, (int)Math.Round(0.14 * _outputSampleRate));
        if (usable <= 0) throw new InvalidDataException("Unexpected RVC audio tensor size.");
        var start = audio.Length - usable;
        var outputLength = (int)Math.Round(usable * 48000d / _outputSampleRate);
        if (outputLength <= 0 || outputLength > _currentOutput.Length)
            throw new InvalidDataException($"Resampled output size ({outputLength}) is not supported.");
        var fade = Math.Min(BoundaryCrossfadeSamples, outputLength);
        for (var i = 0; i < outputLength; i++)
        {
            var sourcePosition = i * (usable - 1d) / Math.Max(1, outputLength - 1);
            var left = Math.Min((int)sourcePosition, usable - 1);
            var right = Math.Min(left + 1, usable - 1);
            var fraction = (float)(sourcePosition - left);
            var value = audioArray[start + left] + (audioArray[start + right] - audioArray[start + left]) * fraction;
            _currentOutput[i] = value;
        }
        if (_hasTail)
        {
            for (var i = 0; i < fade; i++)
            {
                var t = (i + 1f) / fade;
                var alpha = 0.5f * (1f - MathF.Cos(MathF.PI * t));
                var bridge = _previousLastSample + (_currentOutput[fade - 1] - _previousLastSample) * t;
                _currentOutput[i] = bridge * (1f - alpha) + _currentOutput[i] * alpha;
            }
        }
        _previousLastSample = _currentOutput[outputLength - 1];
        _hasTail = true;
        return _currentOutput.AsSpan(0, outputLength);
    }

    public void ResetCrossfade()
    {
        _hasTail = false;
        _previousLastSample = 0f;
    }

    private static int ResolveOutputSampleRate(string modelDirectory, InferenceSession generator)
    {
        foreach (var pair in generator.ModelMetadata.CustomMetadataMap)
        {
            if ((pair.Key.Equals("sample_rate", StringComparison.OrdinalIgnoreCase) ||
                 pair.Key.Equals("sr", StringComparison.OrdinalIgnoreCase)) &&
                int.TryParse(pair.Value.TrimEnd('k', 'K'), out var parsed))
                return parsed < 1000 ? parsed * 1000 : parsed;
        }

        var sidecar = Path.Combine(modelDirectory, "sample_rate.txt");
        if (File.Exists(sidecar) && int.TryParse(File.ReadAllText(sidecar).Trim().TrimEnd('k', 'K'), out var sidecarRate))
            return sidecarRate < 1000 ? sidecarRate * 1000 : sidecarRate;
        return 40000;
    }

    private static void ResampleTo16k(ReadOnlySpan<float> source, Span<float> destination)
    {
        if (source.Length != destination.Length * 3)
            throw new ArgumentException("Expected exactly 3:1 resampling from 48 kHz to 16 kHz.", nameof(source));
        for (var i = 0; i < destination.Length; i++)
        {
            var sourceIndex = i * 3;
            destination[i] = (source[sourceIndex] + source[sourceIndex + 1] + source[sourceIndex + 2]) / 3f;
        }
    }

    private static string FindInput(InferenceSession session, params string[] preferred) =>
        FindOptionalInput(session, preferred) ?? session.InputMetadata.Keys.First();

    private static string FindOutput(InferenceSession session, string name) =>
        session.OutputMetadata.Keys.FirstOrDefault(key => string.Equals(key, name, StringComparison.OrdinalIgnoreCase))
        ?? session.OutputMetadata.Keys.First();

    private static string? FindOptionalOutput(InferenceSession session, string name) =>
        session.OutputMetadata.Keys.FirstOrDefault(key => string.Equals(key, name, StringComparison.OrdinalIgnoreCase));

    private static string? FindOptionalInput(InferenceSession session, params string[] preferred) =>
        session.InputMetadata.Keys.FirstOrDefault(name => preferred.Any(p => string.Equals(name, p, StringComparison.OrdinalIgnoreCase)));

    private static InferenceSession CreateSession(string modelPath)
    {
        if (OperatingSystem.IsLinux() || OperatingSystem.IsWindows())
        {
            try
            {
                using var gpuOptions = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
                gpuOptions.AppendExecutionProvider_CUDA(0);
                return new InferenceSession(modelPath, gpuOptions);
            }
            catch (Exception ex) when (ex is OnnxRuntimeException or DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
            {
                Console.Error.WriteLine($"CUDA is unavailable for '{Path.GetFileName(modelPath)}'; falling back to CPU: {ex.Message}");
            }
        }
        using var cpuOptions = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        return new InferenceSession(modelPath, cpuOptions);
    }

    private void FillGaussianNoise(float[] destination, int count)
    {
        for (var i = 0; i < count; i += 2)
        {
            var u1 = Math.Max(_noiseRandom.NextDouble(), double.Epsilon);
            var u2 = _noiseRandom.NextDouble();
            var magnitude = Math.Sqrt(-2.0 * Math.Log(u1));
            destination[i] = (float)(magnitude * Math.Cos(2.0 * Math.PI * u2));
            if (i + 1 < count) destination[i + 1] = (float)(magnitude * Math.Sin(2.0 * Math.PI * u2));
        }
    }

    public void Dispose()
    {
        _content.Dispose();
        _pitch?.Dispose();
        _generator.Dispose();
    }
}
