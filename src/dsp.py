import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
import librosa
import matplotlib.pyplot as plt

def stft(signal, frame_size, hop_size):
    """
    Short-Time Fourier Transform (STFT)
    :param signal: The input signal
    :param frame_size: Size of each frame
    :param hop_size: Hop size (stride)
    :return: STFT of the signal
    """
    return librosa.stft(signal, n_fft=frame_size, hop_length=hop_size)

def istft(stft_matrix, hop_size):
    """
    Inverse Short-Time Fourier Transform (ISTFT)
    :param stft_matrix: STFT of the signal
    :param hop_size: Hop size (stride)
    :return: Reconstructed signal
    """
    return librosa.istft(stft_matrix, hop_length=hop_size)

def fft(signal):
    """
    Fast Fourier Transform (FFT)
    :param signal: The input signal
    :return: FFT of the signal
    """
    return fftpack.fft(signal)

def ifft(fft_signal):
    """
    Inverse Fast Fourier Transform (IFFT)
    :param fft_signal: FFT of the signal
    :return: Reconstructed signal
    """
    return fftpack.ifft(fft_signal)

def cqt(signal, sr, hop_size):
    """
    Constant-Q Transform (CQT)
    :param signal: The input signal
    :param sr: Sampling rate of the signal
    :param hop_size: Hop size (stride)
    :return: CQT of the signal
    """
    return librosa.cqt(signal, sr=sr, hop_length=hop_size)

def spectrogram(signal, frame_size, hop_size):
    """
    Computing spectrogram of the signal
    :param signal: The input signal
    :param frame_size: Size of each frame
    :param hop_size: Hop size (stride)
    :return: Spectrogram of the signal
    """
    frequencies, times, spectrogram = signal.spectrogram(signal, nperseg=frame_size, noverlap=frame_size - hop_size)
    return frequencies, times, spectrogram

def mel_spectrogram(signal, sr, n_fft, hop_length, n_mels):
    """
    Computing Mel spectrogram of the signal
    :param signal: The input signal
    :param sr: Sampling rate of the signal
    :param n_fft: Number of FFT components
    :param hop_length: Hop length (stride)
    :param n_mels: Number of Mel bands
    :return: Mel spectrogram of the signal
    """
    return librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

def plot_spectrogram(spectrogram, sr, hop_length, y_axis="linear"):
    """
    Plotting the spectrogram
    :param spectrogram: Spectrogram of the signal
    :param sr: Sampling rate
    :param hop_length: Hop length (stride)
    :param y_axis: Type of y-axis ('linear', 'log', 'mel', etc.)
    """
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sr, hop_length=hop_length, y_axis=y_axis, x_axis="time")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.show()

def group_delay(signal, sr):
    """
    Computing group delay of the signal
    :param signal: The input signal
    :param sr: Sampling rate of the signal
    :return: Group delay of the signal
    """
    _, _, phase = signal.stft(signal, fs=sr)
    unwrapped_phase = np.unwrap(np.angle(phase))
    group_delay = -np.diff(unwrapped_phase, axis=-1)
    return group_delay

def instantaneous_frequency(signal, sr):
    """
    Computing instantaneous frequency of the signal
    :param signal: The input signal
    :param sr: Sampling rate of the signal
    :return: Instantaneous frequency of the signal
    """
    analytic_signal = signal.hilbert(signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * sr
    return instantaneous_frequency

def apply_filter(signal, filter_type, cutoff, sr, order=5):
    """
    Applying filter to the signal
    :param signal: The input signal
    :param filter_type: Type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop')
    :param cutoff: Cutoff frequency/frequencies
    :param sr: Sampling rate of the signal
    :param order: Order of the filter
    :return: Filtered signal
    """
    nyquist = 0.5 * sr
    normalized_cutoff = np.array(cutoff) / nyquist

    if filter_type == "lowpass":
        b, a = signal.butter(order, normalized_cutoff, btype="low", analog=False)
    elif filter_type == "highpass":
        b, a = signal.butter(order, normalized_cutoff, btype="high", analog=False)
    elif filter_type == "bandpass":
        b, a = signal.butter(order, normalized_cutoff, btype="band", analog=False)
    elif filter_type == "bandstop":
        b, a = signal.butter(order, normalized_cutoff, btype="bandstop", analog=False)
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")

    return signal.filtfilt(b, a, signal)

if __name__ == "__main__":
    import librosa

    example_signal, sr = librosa.load(librosa.ex('trumpet'))

    # To compute STFT
    stft_matrix = stft(example_signal, frame_size=2048, hop_size=512)
    print("STFT computed")

    frequencies, times, spec = spectrogram(example_signal, frame_size=2048, hop_size=512)
    print("Spectrogram computed")

    plot_spectrogram(spec, sr, hop_length=512)
    print("Spectrogram plotted")

    mel_spec = mel_spectrogram(example_signal, sr, n_fft=2048, hop_length=512, n_mels=128)
    print("Mel Spectrogram computed")
