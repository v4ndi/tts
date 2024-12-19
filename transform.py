from config import hp
import torch
import torchaudio
from torchaudio.functional import spectrogram

symbol_to_id = {
  s: i for i, s in enumerate(hp.symbols)
}

def text_to_seq(text):
    text = text.lower()
    seq = []
    for s in text:
        _id = symbol_to_id.get(s, None)
        if _id is not None:
            seq.append(_id)

    seq.append(symbol_to_id["EOS"])
    return torch.IntTensor(seq)

spec_transform = torchaudio.transforms.Spectrogram(
    n_fft=hp.n_fft, 
    win_length=hp.win_length,
    hop_length=hp.hop_length,
    power=hp.power
)


mel_scale_transform = torchaudio.transforms.MelScale(
  n_mels=hp.mel_freq, 
  sample_rate=hp.sr, 
  n_stft=hp.n_stft
)


mel_inverse_transform = torchaudio.transforms.InverseMelScale(
  n_mels=hp.mel_freq, 
  sample_rate=hp.sr, 
  n_stft=hp.n_stft
).cuda()


griffnlim_transform = torchaudio.transforms.GriffinLim(
    n_fft=hp.n_fft,
    win_length=hp.win_length,
    hop_length=hp.hop_length
).cuda()


def pow_to_db_mel_spec(mel_spec):
    mel_spec = torchaudio.functional.amplitude_to_DB(
        mel_spec,
        multiplier = hp.ampl_multiplier, 
        amin = hp.ampl_amin, 
        db_multiplier = hp.db_multiplier, 
        top_db = hp.max_db
    )
    mel_spec = mel_spec/hp.scale_db
    return mel_spec


def db_to_power_mel_spec(mel_spec):
    mel_spec = mel_spec*hp.scale_db
    mel_spec = torchaudio.functional.DB_to_amplitude(
        mel_spec,
        ref=hp.ampl_ref,
        power=hp.ampl_power
    )  
    return mel_spec


def convert_to_mel_spec(wav):
    spec = spec_transform(wav)
    mel_spec = mel_scale_transform(spec)
    db_mel_spec = pow_to_db_mel_spec(mel_spec)
    db_mel_spec = db_mel_spec.squeeze(0)
    return db_mel_spec


def inverse_mel_spec_to_wav(mel_spec):
    power_mel_spec = db_to_power_mel_spec(mel_spec)
    spectrogram = mel_inverse_transform(power_mel_spec)
    pseudo_wav = griffnlim_transform(spectrogram)
    return pseudo_wav


def mask_from_seq_lengths(
    sequence_lengths: torch.Tensor, 
    max_length: int
) -> torch.BoolTensor:
    """
    our input was `[2, 2, 3]`, with a `max_length` of 4, we'd return
    `[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]`.
    """
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return sequence_lengths.unsqueeze(1) >= range_tensor 