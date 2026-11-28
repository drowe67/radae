#!/usr/bin/env python3
"""99% power occupied bandwidth (OBW) of a RADE signal, e.g. the --write_rx output of inference.py.

Input: rate Fs (8 kHz) complex samples (.f32, ..IQIQ..) or real int16 .wav.
Spectrum: Welch average of Hann windowed FFTs, window length chosen so the
equivalent noise bandwidth equals the resolution bandwidth (RBW, default 10 Hz),
50% overlap. OBW: band containing 99% of the power, excluding 0.5% at each edge.
"""
import argparse, wave, numpy as np

def load(fn, fs=8000):
    if fn.endswith('.wav'):
        w = wave.open(fn); fs = w.getframerate()
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float), fs
    x = np.fromfile(fn, dtype=np.float32)
    return x[::2] + 1j*x[1::2], fs

def obw99(x, fs, rbw=10.0):
    n = int(round(1.5*fs/rbw))
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::n//2]*np.hanning(n)
    P = np.mean(np.abs(np.fft.fft(frames, axis=1))**2, axis=0)
    f = np.fft.fftfreq(n, 1/fs)
    if not np.iscomplexobj(x):                 # real signal: positive frequencies only
        keep = f >= 0; P, f = P[keep], f[keep]
    o = np.argsort(f); P, f = P[o], f[o]
    c = np.cumsum(P)/P.sum()
    lo, hi = np.interp(0.005, c, f), np.interp(0.995, c, f)
    return hi-lo, lo, hi, f, P

def plot_psd(fn, f, P, lo, hi):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 16})
    PdB = 10*np.log10(P/P.max())
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(f, PdB, lw=1)
    ax.axvline(lo, color='r', lw=2, zorder=3); ax.axvline(hi, color='r', lw=2, zorder=3)
    ax.set_xlim(0, 3000); ax.set_xticks([0, 1000, 2000, 3000]); ax.set_ylim(-40, 0); ax.grid()
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('dB')
    fig.tight_layout(); fig.savefig(fn)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('file'); p.add_argument('--rbw', type=float, default=10.0)
    p.add_argument('--plot', help='plot spectrum with OBW edges to this file (e.g. .png, .pdf)')
    a = p.parse_args(); x, fs = load(a.file)
    bw, lo, hi, f, P = obw99(x, fs, a.rbw)
    if a.plot:
        plot_psd(a.plot, f, P, lo, hi)
    print("OBW99: %.1f Hz (%.1f to %.1f Hz) RBW: %.1f Hz" % (bw, lo, hi, a.rbw))
