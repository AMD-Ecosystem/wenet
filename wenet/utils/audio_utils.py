"""Audio I/O that does not depend on torchaudio's I/O backends.

torchaudio 2.9 removed torchaudio.info and the sox effects, and routes
torchaudio.load/save through TorchCodec. soundfile (libsndfile) reads and
writes wav, flac, ogg/opus and mp3; other formats fall back to
torchaudio.load.
"""

import io

import soundfile as sf
import torch
import torchaudio


def load_audio(file, frame_offset: int = 0, num_frames: int = -1):
    """Like torchaudio.load with its defaults: returns a float32 waveform
    of shape (channels, frames), scaled to [-1, 1], and the sample rate.

    Args:
        file: path or file-like object
        frame_offset: number of frames to skip
        num_frames: number of frames to read, -1 for all
    """
    pos = file.tell() if isinstance(file, io.IOBase) else None
    try:
        data, sample_rate = sf.read(file,
                                    frames=num_frames,
                                    start=frame_offset,
                                    dtype='float32',
                                    always_2d=True)
    except RuntimeError:  # soundfile.LibsndfileError
        if pos is not None:
            file.seek(pos)
        return torchaudio.load(file,
                               frame_offset=frame_offset,
                               num_frames=num_frames)
    return torch.from_numpy(data.T.copy()), sample_rate


def audio_sample_rate(file) -> int:
    """Sample rate of an audio file, leaving a file-like object where it
    was."""
    pos = file.tell() if isinstance(file, io.IOBase) else None
    try:
        return sf.info(file).samplerate
    except RuntimeError:  # soundfile.LibsndfileError
        if pos is not None:
            file.seek(pos)
        return load_audio(file)[1]
    finally:
        if pos is not None:
            file.seek(pos)


def save_audio(file, waveform: torch.Tensor, sample_rate: int,
               format: str = 'wav'):
    """Write a (channels, frames) waveform: 16-bit PCM for an int16
    tensor, 32-bit float otherwise (torchaudio.save's defaults for wav)."""
    subtype = 'PCM_16' if waveform.dtype == torch.int16 else 'FLOAT'
    data = waveform.detach().cpu()
    if data.is_floating_point():
        data = data.float()
    sf.write(file, data.numpy().T, sample_rate, format=format,
             subtype=subtype)
