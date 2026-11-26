import glob
import io

import pytest
import torch
import torchaudio

from wenet.utils.audio_utils import load_audio, save_audio


def test_save_load_consistently():
    wav_paths = glob.glob("test/resources/*.wav")
    for wav_path in wav_paths:
        wav, sr = load_audio(wav_path)
        with io.BytesIO() as f:
            wav = torchaudio.transforms.Resample(sr, sr)(wav)
            wav_short = (wav * (1 << 15))
            wav_short = wav_short.to(torch.int16)
            save_audio(f, wav_short, sr, format="wav")
            f.seek(0)
            b = f.read()

        with io.BytesIO(b) as f:
            new_wav, new_sr = load_audio(f)
            assert new_sr == sr
            assert torch.allclose(new_wav, wav)


@pytest.mark.skipif(not hasattr(torchaudio.utils, "sox_utils"),
                    reason="torchaudio >= 2.9 has no sox_utils")
def test_sox_set_buffer():
    torchaudio.utils.sox_utils.set_buffer_size(16500)


def test_make_shards():
    # TODO(MDdct): add make shards
    pass
