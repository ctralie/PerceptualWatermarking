import librosa
import numpy as np
import torch
from torch import nn
from scipy.io import wavfile
import subprocess
import os


def get_mp3_noise(X, sr, diff=True):
    """
    Compute the mp3 noise of a batch of audio samples using ffmpeg
    as a subprocess
    
    Parameters
    ----------
    X: torch.tensor(n_batches, n_samples)
        Audio samples
    sr: int
        Audio sample rate
    diff: bool
        If True, return the difference.  If False, return the mp3 audio
    
    Returns
    -------
    torch.tensor(n_batches, n_samples)
        mp3 noise
    """
    orig_T = X.shape[1]
    X = nn.functional.pad(X, (0, X.shape[1]//4, 0, 0))
    x = X.detach().cpu().numpy().flatten()
    x = np.array(x*32768, dtype=np.int16)
    fileprefix = "temp{}".format(np.random.randint(1000000))
    wavfilename = "{}.wav".format(fileprefix)
    mp3filename = "{}.mp3".format(fileprefix)
    wavfile.write(wavfilename, sr, x)
    if os.path.exists(mp3filename):
        os.remove(mp3filename)
    subprocess.call("ffmpeg -i {} {}".format(wavfilename, mp3filename).split(), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    x, _ = librosa.load(mp3filename, sr=sr)
    os.remove(wavfilename)
    os.remove(mp3filename)
    x = np.reshape(x, X.shape)
    Y = torch.from_numpy(x).to(X)
    if diff:
        Y -= X
    return Y[:, 0:orig_T]

def get_batch_stft(X, win_length, interp_fac=1):
    """
    Perform a Hann-windowed real STFT on batches of audio samples,
    assuming the hop length is half of the window length

    Parameters
    ----------
    X: torch.tensor(n_batches, n_samples)
        Audio samples
    win_length: int
        Window length
    interp_fac: int
        Factor by which to interpolate the frequency bins
    
    Returns
    -------
    S: torch.tensor(n_batches, 1+2*(n_samples-win_length)/win_length), win_length*interp_fac//2+1)
        Real windowed STFT
    """
    n_batches = X.shape[0]
    n_samples = X.shape[1]
    hop_length = win_length//2
    T = (n_samples-win_length)//hop_length+1
    hann = torch.hann_window(win_length).to(X)
    hann = hann.view(1, 1, win_length)

    ## Take out each overlapping window of the signal
    XW = torch.zeros(n_batches, T, win_length*interp_fac).to(X)
    n_even = n_samples//win_length
    XW[:, 0::2, 0:win_length] = X[:, 0:n_even*win_length].view(n_batches, n_even, win_length)
    n_odd = T - n_even
    XW[:, 1::2, 0:win_length] = X[:, hop_length:hop_length+n_odd*win_length].view(n_batches, n_odd, win_length)
    
    # Apply hann window and invert
    XW[:, :, 0:win_length] *= hann
    return torch.fft.rfft(XW, dim=-1)

def get_batch_istft(S, win_length):
    """
    Invert a Hann-windowed real STFT on batches of audio samples,
    assuming the hop length is half of the window length

    Parameters
    ----------
    S: torch.tensor(n_batches, 1+2*(n_samples-win_length)/win_length), win_length//2+1)
        Real windowed STFT
    win_length: int
        Window length
    
    Returns
    -------
    X: torch.tensor(n_batches, n_samples)
        Audio samples
    """
    hop_length = win_length//2
    n_batches = S.shape[0]
    T = S.shape[1]
    n_samples = T*hop_length + win_length - 1
    XInv = torch.fft.irfft(S)
    XEven = XInv[:, 0::2, :].flatten(1, 2)
    XOdd  = XInv[:, 1::2, :].flatten(1, 2)
    X = torch.zeros(n_batches, n_samples).to(XEven)
    X[:, 0:XEven.shape[1]] = XEven
    X[:, hop_length:hop_length+XOdd.shape[1]] += XOdd
    return X


HANN_TABLE = {}
def mss_loss(X, Y, eps=1e-7):
    """
    Compute the multi-scale spectral loss between two sets of audio samples

    Parameters
    ----------
    X: torch.tensor(n_batches, n_samples)
        First set of audio samples
    Y: torch.tensor(n_batches, n_samples)
        Second set of audio samples
    eps: float
        Lower floor for log of spectrogram

    Returns
    -------
    float: MSS loss
    """
    global HANN_TABLE
    loss = 0
    win = 64
    while win <= 2048:
        hop = win//4
        if not win in HANN_TABLE:
            HANN_TABLE[win] = torch.hann_window(win).to(X)
        hann = HANN_TABLE[win]
        SX = torch.abs(torch.stft(X.squeeze(), win, hop, win, hann, return_complex=True))
        SY = torch.abs(torch.stft(Y.squeeze(), win, hop, win, hann, return_complex=True))
        loss_win = torch.sum(torch.abs(SX-SY)) + torch.sum(torch.abs(torch.log(SX+eps)-torch.log(SY+eps)))
        loss += loss_win/torch.numel(SX)
        win *= 2
    return loss
