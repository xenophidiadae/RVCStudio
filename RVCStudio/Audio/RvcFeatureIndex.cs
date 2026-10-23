using System;
using System.IO;
using System.Numerics;

namespace RVCStudio.Audio;

/// <summary>In-memory IVF index with per-channel int8-quantized residual vectors.</summary>
internal sealed class RvcFeatureIndex
{
    private const int HeaderSize = 16;
    private readonly int _dimension;
    private readonly int _clusterCount;
    private readonly float[] _centroids;
    private readonly float[] _scales;
    private readonly long[] _offsets;
    private readonly sbyte[] _codes;

    private RvcFeatureIndex(int dimension, int clusterCount, float[] centroids,
        float[] scales, long[] offsets, sbyte[] codes)
    {
        _dimension = dimension;
        _clusterCount = clusterCount;
        _centroids = centroids;
        _scales = scales;
        _offsets = offsets;
        _codes = codes;
    }

    public static RvcFeatureIndex Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);
        if (reader.ReadUInt32() != 0x49435652) throw new InvalidDataException("Неверная сигнатура RVCI индекса.");
        var dimension = reader.ReadInt32();
        var clusters = reader.ReadInt32();
        var vectors = reader.ReadInt32();
        if (dimension is < 1 or > 4096 || clusters < 1 || vectors < 1 || clusters > vectors)
            throw new InvalidDataException("Некорректные размеры RVCI индекса.");
        var expected = (long)HeaderSize + 4L * dimension * clusters * 2 + 8L * (clusters + 1) + (long)vectors * dimension;
        if (stream.Length != expected) throw new InvalidDataException("Размер файла RVCI не совпадает с заголовком.");
        var valueCount = checked(dimension * clusters);
        var centroids = ReadFloats(reader, valueCount);
        var scales = ReadFloats(reader, valueCount);
        var offsets = new long[clusters + 1];
        Span<byte> offsetBytes = stackalloc byte[sizeof(long)];
        for (var i = 0; i < offsets.Length; i++)
        {
            reader.ReadExactly(offsetBytes);
            offsets[i] = System.Buffers.Binary.BinaryPrimitives.ReadInt64LittleEndian(offsetBytes);
            if (i > 0 && offsets[i] < offsets[i - 1]) throw new InvalidDataException("Немонотонные смещения кластеров RVCI.");
        }
        if (offsets[0] != 0 || offsets[^1] != vectors) throw new InvalidDataException("Некорректные смещения кластеров RVCI.");
        for (var i = 0; i < clusters; i++)
            if (offsets[i] == offsets[i + 1]) throw new InvalidDataException("RVCI индекс содержит пустой кластер.");
        var codes = new sbyte[checked(vectors * dimension)];
        var raw = reader.ReadBytes(codes.Length);
        if (raw.Length != codes.Length) throw new EndOfStreamException();
        Buffer.BlockCopy(raw, 0, codes, 0, raw.Length);
        return new RvcFeatureIndex(dimension, clusters, centroids, scales, offsets, codes);
    }

    public void Blend(float[] features, int frames, int dimension, float indexRate)
    {
        if (indexRate <= 0f) return;
        indexRate = Math.Clamp(indexRate, 0f, 1f);
        if (dimension != _dimension) throw new InvalidDataException($"Размерность индекса ({_dimension}) не совпадает с признаками ({dimension}).");
        var probes = Math.Min(8, _clusterCount);
        Span<int> bestClusters = stackalloc int[8];
        Span<float> bestDistances = stackalloc float[8];
        var simdWidth = Vector<float>.Count;
        Span<float> reconstructedResiduals = stackalloc float[Vector<float>.Count];
        for (var frame = 0; frame < frames; frame++)
        {
            var featureOffset = frame * dimension;
            bestDistances[..probes].Fill(float.PositiveInfinity);
            for (var cluster = 0; cluster < _clusterCount; cluster++)
            {
                var centroidOffset = cluster * dimension;
                var sum = Vector<float>.Zero;
                var c = 0;
                for (; c <= dimension - simdWidth; c += simdWidth)
                {
                    var delta = new Vector<float>(features, featureOffset + c) - new Vector<float>(_centroids, centroidOffset + c);
                    sum += delta * delta;
                }
                var distance = Vector.Sum(sum);
                for (; c < dimension; c++)
                {
                    var delta = features[featureOffset + c] - _centroids[centroidOffset + c];
                    distance += delta * delta;
                }
                if (distance >= bestDistances[probes - 1]) continue;
                var insert = probes - 1;
                while (insert > 0 && distance < bestDistances[insert - 1])
                {
                    bestDistances[insert] = bestDistances[insert - 1];
                    bestClusters[insert] = bestClusters[insert - 1];
                    insert--;
                }
                bestDistances[insert] = distance;
                bestClusters[insert] = cluster;
            }

            var bestVector = -1;
            var bestOwner = -1;
            var nearestDistance = float.PositiveInfinity;
            for (var p = 0; p < probes; p++)
            {
                var cluster = bestClusters[p];
                var clusterStart = _offsets[cluster];
                var centroidOffset = cluster * dimension;
                for (var vector = clusterStart; vector < _offsets[cluster + 1]; vector++)
                {
                    var codeOffset = checked((int)(vector * dimension));
                    var distance = 0f;
                    var c = 0;
                    for (; c <= dimension - simdWidth; c += simdWidth)
                    {
                        for (var lane = 0; lane < simdWidth; lane++)
                            reconstructedResiduals[lane] = _codes[codeOffset + c + lane] * _scales[centroidOffset + c + lane];
                        var queryResidual = new Vector<float>(features, featureOffset + c) - new Vector<float>(_centroids, centroidOffset + c);
                        var delta = queryResidual - new Vector<float>(reconstructedResiduals);
                        distance += Vector.Sum(delta * delta);
                    }
                    for (; c < dimension; c++)
                    {
                        var reconstructed = _centroids[centroidOffset + c] + _codes[codeOffset + c] * _scales[centroidOffset + c];
                        var delta = features[featureOffset + c] - reconstructed;
                        distance += delta * delta;
                    }
                    if (distance < nearestDistance)
                    {
                        nearestDistance = distance;
                        bestVector = codeOffset;
                        bestOwner = cluster;
                    }
                }
            }
            if (bestVector < 0) continue;
            for (var c = 0; c < dimension; c++)
            {
                var ownerOffset = bestOwner * dimension + c;
                var reconstructed = _centroids[ownerOffset] + _codes[bestVector + c] * _scales[ownerOffset];
                var original = features[featureOffset + c];
                features[featureOffset + c] = original * (1f - indexRate) + reconstructed * indexRate;
            }
        }
    }

    private static float[] ReadFloats(BinaryReader reader, int count)
    {
        var result = new float[count];
        var bytes = new byte[sizeof(float)];
        for (var i = 0; i < count; i++)
        {
            reader.ReadExactly(bytes.AsSpan());
            result[i] = BitConverter.Int32BitsToSingle(System.Buffers.Binary.BinaryPrimitives.ReadInt32LittleEndian(bytes));
        }
        return result;
    }
}
