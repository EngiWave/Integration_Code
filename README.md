# Integration code

** The inputs are NOT included **

This project is a comprehensive implementation of an audio signal processing system developed as part of a final assignment for the "Signals and Systems" course. The system processes audio signals from digital microphones, performing various tasks such as noise reduction, speech enhancement, and modulation.

Project Overview

The project is divided into multiple stages, each focusing on a specific aspect of audio processing. The key modules include:

    PDM to PCM Conversion: Converts Pulse-Density Modulated (PDM) signals to Pulse-Code Modulated (PCM) signals.
    DC Removal: Eliminates the DC component from the audio signal.
    Voice Activity Detection: Detects periods of speech in the audio signal.
    Acoustic Gain Control: Stabilizes the volume of the audio signal.
    Short-Time Fourier Transform (STFT): Analyzes and synthesizes the signal in the frequency domain.
    Noise Reduction: Reduces background noise from the audio signal.
    Pitch Estimation: Estimates the fundamental frequency of the speech signal.
    Speech Speed Adjustment: Slows down or speeds up the speech signal.
    Decimation and Interpolation: Downsamples and upsamples the signal.
    Transmission and Reception: Implements single-sideband modulation for transmission.

Files

    main.m: The primary MATLAB script that integrates all the modules and runs the entire signal processing pipeline.
    config.toml: Configuration file that contains parameters for the various processing stages.

License

This project is licensed under the Bar Ilan License.
