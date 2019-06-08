import numpy as np

N = 512
min_value = -100.0 # min value in dB
max_value = 96.0 # max value in dB

def scaled_fft_db(x):
    """ ASSIGNMENT 1:
        a) Compute a 512-point Hann window and use it to weigh the input data.
        b) Compute the DFT of the weighed input, take the magnitude in dBs and
        normalize so that the maximum value is 96dB.
        c) Return the first 257 values of the normalized spectrum

        Arguments:
        x: 512-point input buffer.

        Returns:
        first 257 points of the normalized spectrum, in dBs
    """

    # Your code goes here
    window = compute_hann()
    y = window * x
    y_fft = np.fft.fft(y) / N # the output of the fft normalized
    y_fft = np.abs(y_fft[:(N / 2 + 1)]) # take only the first half of the absolute values
    y_fft = [20 * np.log10(i) if 20 * np.log10(i) >= min_value else min_value for i in y_fft]
    m = np.max(y_fft)
    m = m - max_value
    y_fft = y_fft - m
    
    return y_fft
    
def compute_hann():
    w = np.zeros(N) # initialize window
    n = np.arange(len(w)) 
    w = 0.5 * (1 - np.cos(2 * np.pi * n) / (N-1))
    c = np.sqrt((N-1) / np.sum(w ** 2))
    w = c * w
    
    return w
        
