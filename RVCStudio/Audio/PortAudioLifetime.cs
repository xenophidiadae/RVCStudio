using PortAudioSharp;
using System;

namespace RVCStudio.Audio;
public static class PortAudioLifetime
{
    private static readonly object Sync = new();
    private static bool _initialized;

    public static void EnsureInitialized()
    {
        lock (Sync)
        {
            if (_initialized) return;
            PortAudio.Initialize();
            _initialized = true;
        }
    }

    public static void Shutdown()
    {
        lock (Sync)
        {
            if (!_initialized) return;
            PortAudio.Terminate();
            _initialized = false;
        }
    }
}
