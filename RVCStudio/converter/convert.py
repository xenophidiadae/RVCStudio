import argparse
import math
import os
import sys
import tempfile
import zipfile
import torch
import torch.nn as nn
import struct
import numpy as np
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from rvc_exporter.models import SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsidONNX


def prepare_index(source_path, output_path):
    """Convert a standard FAISS RVC index to RVCI: IVF centroids + per-channel int8 residuals."""
    import faiss

    source = faiss.read_index(source_path)
    if source.ntotal == 0 or source.d <= 0:
        raise ValueError("FAISS index is empty or invalid")
    if source.ntotal < 2:
        raise ValueError("At least two vectors are required to build an IVF index")
    d, n = source.d, source.ntotal
    # RVC indices are often IndexIDMap( IndexIVFFlat ). Enable reconstruction
    # for IVF storage when the wrapper does not provide a direct map.
    base = faiss.downcast_index(source)
    while hasattr(base, "index"):
        base = faiss.downcast_index(base.index)
    if hasattr(base, "make_direct_map"):
        try:
            base.make_direct_map()
        except RuntimeError:
            pass
    vectors = np.empty((n, d), dtype=np.float32)
    try:
        # RVC's standard IndexIDMap stores its wrapped vector index here.
        vectors[:] = source.index.reconstruct_n(0, n)
    except (AttributeError, RuntimeError, TypeError):
        # Some FAISS wrappers do not expose the inner index or range reconstruction.
        for i in range(n):
            vectors[i] = source.reconstruct(i)

    # FAISS RVC indices commonly wrap IVF-Flat in IndexIDMap; recover IVF centroids
    # when available, otherwise build a coarse quantizer with a bounded cluster count.
    if hasattr(base, "nlist") and hasattr(base, "quantizer"):
        nlist = int(base.nlist)
        centroids = np.vstack([base.quantizer.reconstruct(i) for i in range(nlist)]).astype(np.float32)
    else:
        # Match the conventional RVC IVF sizing heuristic: enough lists to keep
        # per-query candidate scans small, but never fewer than the training set supports.
        nlist = min(max(1, int(16 * np.sqrt(n))), max(1, n // 39))
        coarse = faiss.IndexFlatL2(d)
        clustering = faiss.Clustering(d, nlist)
        clustering.niter = 20
        clustering.min_points_per_centroid = 1
        clustering.train(vectors, coarse)
        centroids = np.vstack([coarse.reconstruct(i) for i in range(nlist)]).astype(np.float32)

    coarse = faiss.IndexFlatL2(d)
    coarse.add(centroids)
    _, assignments = coarse.search(vectors, 1)
    assignments = assignments[:, 0]
    # Remove empty centroids so every serialized IVF list is searchable and non-empty.
    used = np.unique(assignments)
    remap = np.full(nlist, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    centroids = centroids[used]
    nlist = len(used)
    assignments = remap[assignments]
    # Per-dimension symmetric int8 residual quantization minimizes quality loss
    # while allowing simple SIMD dot products at runtime.
    scales = np.zeros((nlist, d), dtype=np.float32)
    order = np.argsort(assignments, kind="stable")
    assignments = assignments[order]
    vectors = vectors[order]
    counts = np.bincount(assignments, minlength=nlist).astype(np.int32)
    offsets = np.zeros(nlist + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    ordered = np.empty((n, d), dtype=np.int8)
    for cluster in range(nlist):
        first, last = offsets[cluster], offsets[cluster + 1]
        if first == last:
            continue
        residuals = vectors[first:last] - centroids[cluster]
        scales[cluster] = np.maximum(np.max(np.abs(residuals), axis=0) / 127.0, 1e-8)
        quantized = np.clip(np.rint(residuals / scales[cluster]), -127, 127).astype(np.int8)
        ordered[first:last] = quantized

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temp_path = output_path + ".tmp"
    with open(temp_path, "wb") as out:
        out.write(struct.pack("<4sIII", b"RVCI", d, nlist, n))
        out.write(centroids.astype("<f4").tobytes())
        out.write(scales.astype("<f4").tobytes())
        out.write(offsets.astype("<i8").tobytes())
        out.write(ordered.tobytes())
    os.replace(temp_path, output_path)
    print(f"[+] RVCI index: {n} vectors, {d} dims, {nlist} clusters, int8 residuals")

                                            
                                  
                                            


class ResBlock1(nn.Module):

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=int((kernel_size * d - d) / 2),
                    )
                )
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=int((kernel_size - 1) / 2),
                    )
                )
                for _ in dilation
            ]
        )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = torch.nn.functional.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = torch.nn.functional.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class SineGen(nn.Module):

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def forward(self, f0):
                                    
        with torch.no_grad():
            f0 = f0.transpose(1, 2)                    
            rad = f0 / self.sampling_rate
                                 
            noise = torch.randn_like(rad) * self.noise_std
            rad_step = (rad + noise) * 2 * math.pi
            phase = torch.cumsum(rad_step, dim=1)

                                 
            harmonics = (
                torch.arange(1, self.dim + 1, device=f0.device).float().reshape(1, 1, -1)
            )
            phases = phase * harmonics

            sine = self.sine_amp * torch.sin(phases)
                                             
            sine = sine * (f0 > self.voiced_threshold).float()
            return sine                            


class SourceModuleHnNSF(nn.Module):

    def __init__(self, sampling_rate, harmonic_num=0):
        super().__init__()
        self.sine_gen = SineGen(sampling_rate, harmonic_num)
                                                              
        self.l_linear = nn.Linear(harmonic_num + 1, 1)

    def forward(self, f0):
                                    
        sine_wavs = self.sine_gen(f0)                      
        sine_merge = torch.tanh(self.l_linear(sine_wavs))                    
        return sine_merge.transpose(1, 2)                    

class Generator(nn.Module):

    def __init__(
        self,
        initial_channel,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
        sr=40000,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

                                                                                           
        scale_factor = math.prod(upsample_rates)
        self.f0_upsamp = nn.Upsample(scale_factor=scale_factor, mode="linear")
        self.m_source = SourceModuleHnNSF(sampling_rate=sr)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(ResBlock1(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, f0, g=None):
                                                                                           
        f0_upsampled = self.f0_upsamp(f0.unsqueeze(1))
        har_source = self.m_source(f0_upsampled)

        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = 0
            for j in range(self.num_kernels):
                xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


                                            
                    
                                            


def convert_pth_to_onnx(pth_path, onnx_path):
    print(f"[*] Читаем модель: {os.path.basename(pth_path)}")
    cpt = torch.load(pth_path, map_location="cpu")

    config = cpt.get("config", None)
    if not config:
        print("[-] ОШИБКА: В .pth нет 'config'!")
        return False

                                    
                            
                                                                                     
                                                                                  
                                                                         
                                                              
     
    if len(config) < 18:
        raise ValueError(f"Ожидался полный RVC config из 18 полей, получено {len(config)}")
    resblock = config[9]
    resblock_kernel_sizes = config[10]
    resblock_dilation_sizes = config[11]
    upsample_rates = config[12]
    upsample_initial_channel = config[13]
    upsample_kernel_sizes = config[14]
    spk_embed_dim = config[15]
    gin_channels = config[16]

                                                                      
    sr = cpt.get("sr", config[-1])
    if isinstance(sr, str):
        sr = int(sr.replace("k", "000"))

                         
    inter_channels = config[2]
    hidden_channels = config[3]
    filter_channels = config[4]
    n_heads = config[5]
    n_layers = config[6]
    kernel_size = config[7]
    p_dropout = config[8]

    is_v2 = config[1] == 768 or cpt.get("version", "v1") == "v2"
    feat_dim = 768 if is_v2 else 256

    print(f"[*] Архитектура: {'v2' if is_v2 else 'v1'} | Sampling Rate: {sr}Hz")
    print(f"[*] Upsample rates: {upsample_rates}")

    if not is_v2:
        raise ValueError("Этот exporter поддерживает только RVC v2 checkpoints.")

    net_g = SynthesizerTrnMs768NSFsid(
        spec_channels=feat_dim,
        segment_size=config[1],
        inter_channels=inter_channels,
        hidden_channels=hidden_channels,
        filter_channels=filter_channels,
        n_heads=n_heads,
        n_layers=n_layers,
        kernel_size=kernel_size,
        p_dropout=p_dropout,
        resblock=resblock,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        upsample_rates=upsample_rates,
        upsample_initial_channel=upsample_initial_channel,
        upsample_kernel_sizes=upsample_kernel_sizes,
        spk_embed_dim=spk_embed_dim,
        gin_channels=gin_channels,
        sr=sr,
        is_half=False,
    )

    state_dict = cpt.get("weight", cpt)
    inference_model = SynthesizerTrnMs768NSFsidONNX(net_g)
    inference_state = {key: value for key, value in state_dict.items() if not key.startswith("enc_q.")}
    incompatible = inference_model.load_state_dict(inference_state, strict=False)
    missing = incompatible.missing_keys
    unexpected = incompatible.unexpected_keys
    required = (
        "enc_p.emb_phone.weight", "enc_p.emb_pitch.weight", "enc_p.proj.weight",
        "dec.conv_pre.weight", "dec.conv_post.weight", "emb_g.weight",
    )
    if missing or unexpected or any(key not in state_dict for key in required):
        raise ValueError(f"Checkpoint/export architecture mismatch. missing={missing[:12]}, unexpected={unexpected[:12]}")
    inference_model.eval()

                                                          
    def clean_weight_norm(m):
        try:
            remove_weight_norm(m)
        except (ValueError, RuntimeError):
            pass

    inference_model.apply(clean_weight_norm)

    test_len = 100
    phone = torch.randn(1, test_len, feat_dim)
    phone_lengths = torch.LongTensor([test_len])
    pitch = torch.full((1, test_len), 128, dtype=torch.long)
    pitchf = torch.full((1, test_len), 180.0)
    ds = torch.LongTensor([0])
    rnd = torch.randn(1, 192, test_len)

    print("[*] Экспорт графа в ONNX (подожди немного)...")
    with torch.no_grad():
        validation_output = inference_model(phone, phone_lengths, pitch, pitchf, ds, rnd)
        validation_rms = validation_output.float().square().mean().sqrt().item()
        validation_peak = validation_output.float().abs().max().item()
        print(f"[*] PyTorch sanity: rms={validation_rms:.6f}, peak={validation_peak:.6f}")
        if not torch.isfinite(validation_output).all() or validation_rms < 1e-4 or validation_peak < 1e-3:
            raise ValueError("PyTorch export sanity check failed: generator output is non-finite or near silent.")
        torch.onnx.export(
            inference_model,
            (phone, phone_lengths, pitch, pitchf, ds, rnd),
            onnx_path,
            input_names=["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"],
            output_names=["audio"],
            dynamic_axes={
                "phone": {1: "time"},
                "phone_lengths": {0: "batch"},
                "pitch": {1: "time"},
                "pitchf": {1: "time"},
                "rnd": {2: "time"},
                "audio": {2: "audio_len"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.helper.set_model_props(onnx_model, {"sample_rate": str(sr), "rvc_export": "official-v2-inference"})
    onnx.save(onnx_model, onnx_path)
    print(f"[+] УСПЕШНО сохранен ONNX: {onnx_path}")
    return True


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--prepare-index":
        prepare_index(sys.argv[2], sys.argv[3])
        return
    parser = argparse.ArgumentParser(
        description="Конвертер RVC zip в ONNX без зависимостей"
    )
    parser.add_argument("zip_path", type=str, help="Путь к .zip архиву с моделью")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Куда сохранить (дефолт: рядом с zip)",
    )
    args = parser.parse_args()

    zip_path = os.path.expanduser(args.zip_path)
    if not os.path.exists(zip_path):
        print(f"[-] Файл не существует: {zip_path}")
        sys.exit(1)

    out_dir = (
        os.path.expanduser(args.output)
        if args.output
        else os.path.dirname(os.path.abspath(zip_path))
    )
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(zip_path))[0]

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"[*] Распаковка архива во временную директорию...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp_dir)

        pth_files, index_files = [], []
        for root, _, files in os.walk(tmp_dir):
            for f in files:
                if f.lower().endswith(".pth"):
                    pth_files.append(os.path.join(root, f))
                elif f.lower().endswith(".index"):
                    index_files.append(os.path.join(root, f))

        if not pth_files:
            print("[-] ОШИБКА: В архиве нет файла .pth!")
            sys.exit(1)
        if len(pth_files) > 1:
            print("[-] ОШИБКА: В архиве несколько .pth файлов. Оставьте один файл модели.")
            sys.exit(1)

        onnx_path = os.path.join(out_dir, f"{base_name}.onnx")
        if convert_pth_to_onnx(pth_files[0], onnx_path):
            if index_files:
                dst_index = os.path.join(out_dir, "features.rvci")
                prepare_index(index_files[0], dst_index)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
