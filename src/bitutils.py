import numpy as np
import torch

def get_binary_encoding(Y, bits_per_channel):
    """
    Parameters
    ----------
    Y: torch.tensor(n_batches, time, len(bits_per_channel))
        Tensor to encode, with values in [0, 1]
    bits_per_channel: list of num_channels ints
        Number of bits to use in each channel
    
    Returns
    -------
    torch.tensor(n_batches, time, sum(bits_per_channel))
        Binary encoding of all channels
    """
    total_bits = np.sum(bits_per_channel)
    YBin = torch.zeros(Y.shape[0], Y.shape[1], total_bits, dtype=Y.dtype)
    YBin = YBin.to(Y)
    bit_offset = 0
    for dim, bits in enumerate(bits_per_channel):
        Ydim = (2**bits-1)*Y[:, :, dim]
        place = 2**(bits-1)
        while place > 0:
            bit = 1.0*(Ydim-place > 0)
            YBin[:, :, bit_offset] = bit
            Ydim -= bit*place
            place = place // 2
            bit_offset += 1
    return YBin

def decode_binary(YBin, bits_per_channel, normalize=True):
    """
    Parameters
    ----------
    YBin: torch.tensor(n_batches, time, sum(bits_per_channel))
        Tensor to encode, with values in [0, 1]
    bits_per_channel: list of num_channels ints
        Number of bits to use in each channel
    normalize: bool
        If True, normalize each channel to [0, 1]
    
    Returns
    -------
    torch.tensor(n_batches, time, len(bits_per_channel))
        Binary encoding of all channels
    """
    Y = torch.zeros(YBin.shape[0], YBin.shape[1], len(bits_per_channel), dtype=YBin.dtype)
    Y = Y.to(YBin)
    bit_offset = 0
    for dim, bits in enumerate(bits_per_channel):
        Ydim = (2**(bits-1))*YBin[:, :, bit_offset]
        place = 2**(bits-2)
        bit_offset += 1
        while place >= 1:
            Ydim += place*YBin[:, :, bit_offset]
            place = place // 2
            bit_offset += 1
        if normalize:
            Y[:, :, dim] = Ydim/(2**bits-1)
        else:
            Y[:, :, dim] = Ydim
    return Y

def get_gray_codes(n_bits):
    """
    Compute gray code sequences of a specified size
    
    Parameters
    ----------
    n_bits: int
        Number of bits to use
    
    Returns
    -------
    list of lists of n_bits 1's and 0's for gray codes
    """
    res = []
    if n_bits == 1:
        res = [[0], [1]]
    else:
        g = get_gray_codes(n_bits-1)
        res = [[0]+b for b in g] + [[1]+b for b in g[::-1]]
    return res

def encode_gray_angle(B, n_bits):
    """
    Convert a binary sequence into a gray code sequence represented
    by floats in [0, 2*pi]
    
    Parameters
    ----------
    B: ndarray(N)
        An array of bits
    n_bits: int
        Number of bits in the gray code
    
    Returns
    ndarray(ceil(N/n_bits))
        Gray code encoding
    """
    code = get_gray_codes(n_bits)
    code = {tuple(c):2*np.pi*(i+0.5)/2**n_bits for i, c in enumerate(code)}
    N = int(np.ceil(len(B)/n_bits))
    encoding = np.zeros(N)
    for i in range(N):
        c = B[i*n_bits:(i+1)*n_bits]
        if len(c) < n_bits:
            c = c + [0]*(n_bits-len(c)) # Zeropad
        encoding[i] = code[tuple(c)]
    return encoding
    
def decode_gray_angle(x, n_bits):
    """
    Parameters
    ----------
    x: ndarray(N)
        Gray code angle sequence
    n_bits: int
        Number of bits in the gray code
    
    Returns
    -------
    ndarray(N*n_bits)
        Decoded bits
    """
    code = np.array(get_gray_codes(n_bits), dtype=int)
    idx = np.array(np.floor(x*(2**n_bits)/(2*np.pi)), dtype=int)
    idx = idx % (2**n_bits)
    return code[idx].flatten()
