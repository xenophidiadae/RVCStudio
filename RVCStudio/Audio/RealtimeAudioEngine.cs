using PortAudioSharp;
using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Diagnostics;

namespace RVCStudio.Audio;

public sealed class RealtimeAudioEngine : IDisposable
{
    private readonly record struct AudioBlock(float[] Samples, int Length);
    private readonly record struct AudioInputBlock(float[] Samples, bool ReturnToPool, bool IsSilent);
    private const int Hop16k = 2240;
    private const int Hop48k = Hop16k * 3;
    private const int Window48k = 4800 * 3;
    private const int Context48k = Window48k - Hop48k;
    private const int QueueCapacity = 4;
    private const int OutputRecoveryFadeFrames = 240;
    private const int NoiseGateRampFrames = 960;
    private const float NoiseGateOpenThreshold = 0.0015f;
    private const float NoiseGateCloseThreshold = 0.0008f;
    private readonly ConcurrentQueue<float[]> _freeInputBuffers = new();
    private readonly ConcurrentQueue<float[]> _freeOutputBuffers = new();
    private readonly float[] _captureWindow = new float[Window48k];
    private readonly CancellationTokenSource _cancellation = new();
    private readonly AutoResetEvent _hasDataEvent = new(false);
    private readonly RvcOnnxPipeline _pipeline;
    private readonly Stream _stream;
    private readonly Task _worker;
    private readonly float[] _captureHop = new float[Hop48k];
    private readonly AudioInputBlock[] _inputQueueStorage = new AudioInputBlock[QueueCapacity + 1];
    private readonly AudioBlock[] _outputQueueStorage = new AudioBlock[QueueCapacity + 1];
    private readonly float[] _convertOutput = new float[12000];
    private int _inputHead, _inputTail;
    private int _outputHead, _outputTail;
    private bool _wasNoiseGated;
    private int _captureOffset;
    private float[]? _currentOutput;
    private int _outputOffset;
    private int _currentOutputLength;
    private int _outputRecoveryFadeRemaining;
    private float _outputGateGain;
    private int _noiseGateOpen;
    private int _pitchShift;
    private Exception? _workerError;
    private long _inputCallbacks;
    private long _inputBlocksQueued;
    private long _outputUnderrunFrames;
    private long _outputUnderrunCallbacks;
    private long _inferenceBlocks;
    private long _lastInferenceMilliseconds;
    private float _lastInputRms;
    private float _lastOutputRms;
    private double _inputEnergyForHop;
    private double _outputEnergyForHop;
    private int _inputDiagnosticFramesForHop;
    private int _outputDiagnosticFramesForHop;

    public RealtimeAudioEngine(string modelDirectory, string systemModelsDirectory, int inputDevice, int outputDevice)
    {
        PortAudioLifetime.EnsureInitialized();
        _pipeline = new RvcOnnxPipeline(modelDirectory, systemModelsDirectory);
        for (var i = 0; i < QueueCapacity + 1; i++) _freeInputBuffers.Enqueue(new float[Window48k]);
        for (var i = 0; i < QueueCapacity + 1; i++) _freeOutputBuffers.Enqueue(new float[12000]);
        Array.Clear(_captureWindow);
        var input = new StreamParameters
        {
            device = inputDevice,
            channelCount = 1,
            sampleFormat = SampleFormat.Float32,
            suggestedLatency = PortAudio.GetDeviceInfo(inputDevice).defaultLowInputLatency
        };
        var output = new StreamParameters
        {
            device = outputDevice,
            channelCount = 1,
            sampleFormat = SampleFormat.Float32,
            suggestedLatency = PortAudio.GetDeviceInfo(outputDevice).defaultLowOutputLatency
        };
        _stream = new Stream(input, output, 48000, PortAudio.FramesPerBufferUnspecified,
            StreamFlags.ClipOff, AudioCallback, this);
        _worker = Task.Factory.StartNew(WorkerLoop, TaskCreationOptions.LongRunning);
    }

    public Exception? WorkerError => _workerError;
    public bool IsRunning { get; private set; }
    public long InputCallbacks => Interlocked.Read(ref _inputCallbacks);
    public long InputBlocksQueued => Interlocked.Read(ref _inputBlocksQueued);
    public long OutputUnderrunFrames => Interlocked.Read(ref _outputUnderrunFrames);
    public long OutputUnderrunCallbacks => Interlocked.Read(ref _outputUnderrunCallbacks);
    public long InferenceBlocks => Interlocked.Read(ref _inferenceBlocks);
    public long LastInferenceMilliseconds => Interlocked.Read(ref _lastInferenceMilliseconds);
    public float LastInputRms => Volatile.Read(ref _lastInputRms);
    public float LastOutputRms => Volatile.Read(ref _lastOutputRms);
    public int PitchShiftSemitones { get => _pitchShift; set => Volatile.Write(ref _pitchShift, value); }

    public void Start()
    {
        _stream.Start();
        IsRunning = true;
    }

    private static StreamCallbackResult AudioCallback(IntPtr input, IntPtr output, uint frameCount,
        ref StreamCallbackTimeInfo timeInfo, StreamCallbackFlags statusFlags, IntPtr userDataPtr)
    {
        var engine = Stream.GetUserData<RealtimeAudioEngine>(userDataPtr);
        engine.ProcessCallback(input, output, checked((int)frameCount));
        return StreamCallbackResult.Continue;
    }

    private void ProcessCallback(IntPtr input, IntPtr output, int frames)
    {
        var underrunFrames = 0;
        var hopEnergy = 0d;
        for (var i = 0; i < frames; i++)
        {
            var inputBits = input == IntPtr.Zero ? 0 : Marshal.ReadInt32(input, i * sizeof(float));
            var inputSample = input == IntPtr.Zero ? 0 : BitConverter.Int32BitsToSingle(inputBits);
            _inputEnergyForHop += inputSample * inputSample;
            _inputDiagnosticFramesForHop++;
            _captureHop[_captureOffset] = inputSample;
            hopEnergy += inputSample * inputSample;
            _captureOffset++;
            if (_captureOffset == Hop48k)
            {
                var inputHopRms = (float)Math.Sqrt(hopEnergy / Hop48k);
                var wasOpen = Volatile.Read(ref _noiseGateOpen) != 0;
                var gateOpen = wasOpen
                    ? inputHopRms >= NoiseGateCloseThreshold
                    : inputHopRms >= NoiseGateOpenThreshold;
                var isSilent = !gateOpen;
                Volatile.Write(ref _noiseGateOpen, gateOpen ? 1 : 0);
                Array.Copy(_captureWindow, Hop48k, _captureWindow, 0, Context48k);
                Array.Copy(_captureHop, 0, _captureWindow, Context48k, Hop48k);
                Volatile.Write(ref _lastInputRms, (float)Math.Sqrt(_inputEnergyForHop / Math.Max(1, _inputDiagnosticFramesForHop)));
                _inputEnergyForHop = 0;
                _inputDiagnosticFramesForHop = 0;
                if (_freeInputBuffers.TryDequeue(out var completed))
                {
                    Array.Copy(_captureWindow, completed, Window48k);
                    var nextTail = (_inputTail + 1) % _inputQueueStorage.Length;
                    if (nextTail != Volatile.Read(ref _inputHead))
                    {
                        _inputQueueStorage[_inputTail] = new AudioInputBlock(completed, true, isSilent);
                        Volatile.Write(ref _inputTail, nextTail);
                        Interlocked.Increment(ref _inputBlocksQueued);
                        _hasDataEvent.Set();
                    }
                    else _freeInputBuffers.Enqueue(completed);
                }
                else
                {
                }
                _captureOffset = 0;
                hopEnergy = 0;
            }

            float sample;
            var outputGateOpen = Volatile.Read(ref _noiseGateOpen) != 0;
            if (!outputGateOpen && _outputGateGain <= 0f)
            {
                if (!_wasNoiseGated)
                {
                    if (_currentOutput is not null) _freeOutputBuffers.Enqueue(_currentOutput);
                    _currentOutput = null;
                    _currentOutputLength = 0;
                    while (TryDequeueOutput(out var stale)) _freeOutputBuffers.Enqueue(stale.Samples);
                    _wasNoiseGated = true;
                }
                Marshal.WriteInt32(output, i * sizeof(float), 0);
                continue;
            }
            _wasNoiseGated = !outputGateOpen;
            if (_currentOutput is null || _outputOffset >= _currentOutputLength)
            {
                if (TryDequeueOutput(out var next))
                {
                    if (_currentOutput is not null) _freeOutputBuffers.Enqueue(_currentOutput);
                    _currentOutput = next.Samples;
                    _currentOutputLength = next.Length;
                    _outputOffset = 0;
                }
                else
                {
                    underrunFrames++;
                    _outputRecoveryFadeRemaining = OutputRecoveryFadeFrames;
                    sample = 0;
                    Marshal.WriteInt32(output, i * sizeof(float), BitConverter.SingleToInt32Bits(sample));
                    continue;
                }
            }
            sample = _currentOutput[_outputOffset++];
            var gateStep = 1f / NoiseGateRampFrames;
            _outputGateGain = outputGateOpen
                ? Math.Min(1f, _outputGateGain + gateStep)
                : Math.Max(0f, _outputGateGain - gateStep);
            sample *= _outputGateGain;
            if (_outputRecoveryFadeRemaining > 0)
            {
                var progressed = OutputRecoveryFadeFrames - _outputRecoveryFadeRemaining + 1;
                sample *= (float)progressed / OutputRecoveryFadeFrames;
                _outputRecoveryFadeRemaining--;
            }
            _outputEnergyForHop += sample * sample;
            _outputDiagnosticFramesForHop++;
            Marshal.WriteInt32(output, i * sizeof(float), BitConverter.SingleToInt32Bits(sample));
            if (_outputOffset == _currentOutputLength)
            {
                _freeOutputBuffers.Enqueue(_currentOutput);
                _currentOutput = null;
                _currentOutputLength = 0;
            }
        }
        Interlocked.Increment(ref _inputCallbacks);
        if (underrunFrames > 0)
        {
            Interlocked.Add(ref _outputUnderrunFrames, underrunFrames);
            Interlocked.Increment(ref _outputUnderrunCallbacks);
        }
        if (_outputDiagnosticFramesForHop >= Hop48k)
        {
            Volatile.Write(ref _lastOutputRms, (float)Math.Sqrt(_outputEnergyForHop / _outputDiagnosticFramesForHop));
            _outputEnergyForHop = 0;
            _outputDiagnosticFramesForHop = 0;
        }
    }

    private void WorkerLoop()
    {
        try
        {
            while (true)
            {
                if (!TryDequeueInput(out var block))
                {
                    if (_cancellation.IsCancellationRequested) break;
                    _hasDataEvent.WaitOne();
                    continue;
                }
                if (_cancellation.IsCancellationRequested)
                {
                    if (block.ReturnToPool) _freeInputBuffers.Enqueue(block.Samples);
                    break;
                }
                while (TryDequeueInput(out var newer))
                {
                    if (block.ReturnToPool) _freeInputBuffers.Enqueue(block.Samples);
                    block = newer;
                }

                if (block.IsSilent)
                {
                    _pipeline.ResetCrossfade();
                    if (block.ReturnToPool) _freeInputBuffers.Enqueue(block.Samples);
                    continue;
                }

                var inferenceTimer = Stopwatch.StartNew();
                var converted = _pipeline.Convert(block.Samples, Volatile.Read(ref _pitchShift));
                inferenceTimer.Stop();
                Interlocked.Exchange(ref _lastInferenceMilliseconds, inferenceTimer.ElapsedMilliseconds);
                Interlocked.Increment(ref _inferenceBlocks);
                if (!_freeOutputBuffers.TryDequeue(out var result))
                {
                    if (block.ReturnToPool) _freeInputBuffers.Enqueue(block.Samples);
                    continue;
                }
                var length = converted.Length;
                converted.CopyTo(_convertOutput);
                Array.Copy(_convertOutput, result, length);
                if (Volatile.Read(ref _noiseGateOpen) == 0)
                {
                    _pipeline.ResetCrossfade();
                    _freeOutputBuffers.Enqueue(result);
                    if (block.ReturnToPool) _freeInputBuffers.Enqueue(block.Samples);
                    continue;
                }
                if (!TryEnqueueOutput(new AudioBlock(result, length)))
                    _freeOutputBuffers.Enqueue(result);
                if (block.ReturnToPool) _freeInputBuffers.Enqueue(block.Samples);
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex) { _workerError = ex; }
    }

    private bool TryDequeueInput(out AudioInputBlock block)
    {
        var head = _inputHead;
        if (head == Volatile.Read(ref _inputTail)) { block = default; return false; }
        block = _inputQueueStorage[head];
        _inputQueueStorage[head] = default;
        Volatile.Write(ref _inputHead, (head + 1) % _inputQueueStorage.Length);
        return true;
    }

    private bool TryEnqueueOutput(AudioBlock block)
    {
        var tail = _outputTail;
        var nextTail = (_outputTail + 1) % _outputQueueStorage.Length;
        if (nextTail == Volatile.Read(ref _outputHead)) return false;
        _outputQueueStorage[tail] = block;
        Volatile.Write(ref _outputTail, nextTail);
        return true;
    }

    private bool TryDequeueOutput(out AudioBlock block)
    {
        var head = _outputHead;
        if (head == Volatile.Read(ref _outputTail)) { block = default; return false; }
        block = _outputQueueStorage[head];
        _outputQueueStorage[head] = default;
        Volatile.Write(ref _outputHead, (head + 1) % _outputQueueStorage.Length);
        return true;
    }

    public void Dispose()
    {
        _cancellation.Cancel();
        _hasDataEvent.Set();
        if (IsRunning) _stream.Stop();
        _stream.Dispose();
        try { _worker.Wait(); } catch { }
        while (TryDequeueInput(out var pending))
            if (pending.ReturnToPool) _freeInputBuffers.Enqueue(pending.Samples);
        _pipeline.Dispose();
        _hasDataEvent.Dispose();
        _cancellation.Dispose();
    }
}
