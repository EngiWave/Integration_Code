function main
% Load the TOML configuration file using the add-on function
config = toml.read('config.toml');

% clear the outputs directory according to the config
if config('clear_outputs')
    output_dir = 'outputs';
    if exist(output_dir, 'dir')
        % directory exists, try to delete it
        try
            rmdir(output_dir, 's');
            disp('Existing outputs directory deleted.');
        catch
            warning('Failed to delete existing outputs directory. It may be in use or you may not have sufficient permissions.');
        end
    else
        % directory doesn't exist, inform about it
        disp('Outputs directory does not exist. Will create a new one.');
    end
    
    % Try to create the new directory
    try
        mkdir(output_dir);
        disp('New outputs directory created.');
    catch
        error('Failed to create new outputs directory. Check your permissions or if a file with the same name exists.');
    end
end

pdm2pcm(config('Pdm2pcm'));

DC_Remove(config('DC_Remove'));

VAD(config('VAD'));

AGC(config('AGC'));

STFT(config('STFT'));

noise_reduction(config('Noise_Reduction'));

pitch_estimation(config('Pitch_Estimation'));

Audio_Speed_Modification(config('ASM'));

decimation_antialiasing(config('Decimation_Interpolation'));

SSB(config('SSB'));
end


function pdm2pcm(config_segment)
% Step 1: Read the PDM signal from the text file

fileID = fopen(config_segment('input_path'), 'r');

% Check if the file was opened successfully
if fileID == -1
    error('File could not be opened. Check the file path and name.');
end

% Read the PDM data line by line as numeric data
pdmSignal = fscanf(fileID, '%f');
fclose(fileID);

% Define the sampling frequency of the PDM signal
fs = 3.072e6; % PDM sampling frequency (e.g., 3.072 MHz)

% Remove DC offset
pdmSignal = pdmSignal - mean(pdmSignal);

% Step 2: Design a low-pass FIR filter for decimation
decimationFactor = 64; % Decimation factor
filterOrder = 160; % Filter order
cutoffFrequency = 0.4 / decimationFactor; % Cutoff frequency
lowPassFilter = designfilt('lowpassfir', 'FilterOrder', filterOrder, ...
    'CutoffFrequency', cutoffFrequency, 'SampleRate', fs);

% Step 3: Filter the PDM signal
filteredSignal = filter(lowPassFilter, pdmSignal); % Filter the signal

% Step 4: Downsample the filtered signal
pcmSignal = downsample(filteredSignal, decimationFactor);
pcmFs = fs / decimationFactor; % New sampling frequency

% Step 5: Normalize the PCM signal
maxVal = max(abs(pcmSignal));
if maxVal > 0
    pcmSignal = pcmSignal / maxVal; % Normalize to -1 to 1 range
end

% Step 6: Write the PCM signal to a WAV file
audiowrite(config_segment('output_path'), pcmSignal, pcmFs);

% Play the sound using MATLAB
sound(pcmSignal, pcmFs);
end

function DC_Remove(config_segment)
% Reading the WAV file
[signal, Fs] = audioread(config_segment('input_path'));  %% Reading the WAV file and receiving the signal and sampling frequency
sound(signal, Fs);  % Play the original signal

%parameters
%config_segment('buffer_size') = 10000;  % Buffer size for processing (can be adjusted as needed), i.e., how many samples are processed in each segment
config_segment('overlap') = 0.5;  % Overlap between segments (percentage), i.e., what portion of the previous segment overlaps with the new segment
hop_size = floor(config_segment('buffer_size') * (1 - config_segment('overlap')));  %  Hop size, i.e., how many samples are moved forward each time
high_pass_freq = 100;    % High-pass filter frequency (in Hz), i.e., the cutoff frequency above which the signal is filtered 

% Creating plots
figure;

subplot(2,2,1);
input_plot = plot(NaN, NaN);  % Create an empty plot for the input signal
title('Input Signal');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2,2,2);
output_plot = plot(NaN, NaN); % Create an empty plot for the output signal
title('Output Signal (DC Removed)');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2,2,3);
input_spectrum_plot = plot(NaN, NaN);  % Create an empty plot for the input signal spectrum
title('Input Signal Spectrum');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 500])

subplot(2,2,4);
output_spectrum_plot = plot(NaN, NaN); % Create an empty plot for the output signal spectrum
title('Output Signal Spectrum (DC Removed)');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 500]);

% Creating a high-pass filter in the frequency domain
f_axis = Fs * (0:config_segment('buffer_size')-1) / config_segment('buffer_size'); % Create the frequency axis according to the sampling rate and buffer size
high_pass_filter = (f_axis > high_pass_freq); % Define the high-pass filter, every frequency above high_pass_freq gets a 'true' value

% Initializing buffers
num_buffers = floor((length(signal) - config_segment('buffer_size')) / hop_size) + 1;  % Calculate the number of segments to process
x_buffer = zeros(config_segment('buffer_size'), 1);  % % Create an empty buffer for the input signal
y_buffer = zeros(config_segment('buffer_size'), 1); % Create an empty buffer for the output signal
output_signal = zeros(size(signal));   % Create an empty vector for the processed signal

% Main loop for real-time style processing
for i = 1:num_buffers
    % Extracting a segment from the signal
    idx_start = (i-1) * hop_size + 1;    % Calculate the start index of the segment
    idx_end = idx_start + config_segment('buffer_size') - 1;   % Calculate the end index of the segment
    x_buffer = signal(idx_start:idx_end);  % Extract the segment from the original signal

    % Applying FFT
    X = fft(x_buffer); % Perform Fast Fourier Transform on the segment
    
    % Applying high-pass filter
    X_filtered = X .* high_pass_filter';    % Filter the low frequencies by multiplying with the high-pass filter


    % Applying IFFT
    y_buffer = real(ifft(X_filtered));  % Perform Inverse Fast Fourier Transform on the filtered segment
    
    % Save the processed signal to the overall output
    
    output_signal(idx_start:idx_end) = y_buffer;  % Save the processed segment to the overall output


    % Update plots
    set(input_plot, 'XData', (idx_start:idx_end) / Fs, 'YData', x_buffer);  % Update the input signal plot
    set(output_plot, 'XData', (idx_start:idx_end) / Fs, 'YData', y_buffer); % Update the output signal plot
    set(input_spectrum_plot, 'XData', f_axis, 'YData', abs(X));   % Update the input signal spectrum plot
    set(output_spectrum_plot, 'XData', f_axis, 'YData', abs(X_filtered));  % Update the output signal spectrum plot


    % Set Y-axis limits  
    max_y = max(max(abs([X; X_filtered])));  % Calculate the upper limit of the Y-axis according to the spectrum values
    ylim(subplot(2,2,3), [0 max_y]);  % Set the Y-axis limit for the input signal spectrum
    ylim(subplot(2,2,4), [0 max_y]);  % Set the Y-axis limit for the output signal spectrum

    % Pause to simulate real-time processing (remove if not processing in real-time)
 
    pause(config_segment('buffer_size') / Fs);
end

% Save the filtered signal to a new WAV file
audiowrite(config_segment('output_path'), output_signal, Fs);   % Save the filtered signal to a new WAV file

% Play the processed signal after filtering
sound(output_signal, Fs); % Play the filtered signal
end

function VAD(config_segment)
% Function to smooth the speech
function Output_Signal = smooth_speech(Input_Signal)
    % Smoothing the input signal to fill short gaps
    Output_Signal = Input_Signal;
    maxGapLength = 25;  % Gap length (changeable)
    
    for i = 1:length(Input_Signal)
        if Input_Signal(i) == 0
            % Detecting a zero in a proximity of "1"s
            if i > 1 && i < length(Input_Signal)
                if sum(Input_Signal(i-min(i-1,maxGapLength):i-1)) > 0 && sum(Input_Signal(i+1:min(i+maxGapLength,length(Input_Signal)))) > 0
                    Output_Signal(i) = 1;
                end
            end
        end
    end
end


%{
   We created a variable that collects data from the given audio,
   Such as filename, path, total number of samples, duration.
%}
info = audioinfo(config_segment('input_path'));

%{
   Function that reads an audio file and returns two main outputs:
   1. 'samples' contains the total audio samples from the file.
   2. 'sampling_rate'
%}
[samples, sampling_rate] = audioread(config_segment('input_path'));

% Defining parameters
% time_frame = 0.02;       % Parameter chosen according to article 20ms
% time_step = 0.01;        % Parameter chosen according to article 10ms
sample_frame = round(config_segment('time_frame') * sampling_rate);  % Number of samples for each frame
sample_step = round(config_segment('time_frame') * sampling_rate);    % Number of samples for each step
numFrames = floor((length(samples) - sample_frame) / sample_step) + 1;  % Total number of frames

% Calculate the energy of each frame
frame_energy = zeros(1, numFrames);
for i = 1:numFrames
    startIdx = (i - 1) * sample_step + 1;  % Starting index of the frame
    endIdx = startIdx + sample_frame - 1;  % Ending index of the frame
    frame = samples(startIdx:endIdx);  % Extracting the frame from the signal
    frame_energy(i) = sum(frame.^2);  % Calculating the energy of the frame
end

% Setting a threshold according to "deep learning"
energyThreshold = 7;

% Creating a vector for speech detection
Speech = zeros(1, numFrames);

% Checking the conditions for energy threshold (changeable)
Speech(frame_energy > energyThreshold) = 1;

% Filling short gaps using the smooth function
smoothed_signal = smooth_speech(Speech);

% Converting smoothed signal back to binary output
binary_output = zeros(1, numFrames);
binary_output(smoothed_signal > 0.5) = 1;
audiowrite(config_segment('output_path'), binary_output, sampling_rate);
% Plotting binary speech detection result

% Plotting audio signal
timeAxis = (0:length(samples)-1) / sampling_rate;  % Defining time axis
figure;
subplot(2, 1, 1);
plot(timeAxis, samples);
title('Audio Signal');
xlabel('Time (s)');
ylabel('Amplitude');

% The plot of speech
subplot(2, 1, 2);
frameTimeAxis = (0:numFrames-1) * config_segment('time_frame');  % Defining time axis
stairs(frameTimeAxis, binary_output, 'r', 'LineWidth', 2);  
title('Smoothed Binary Speech Presence');
xlabel('Time (s)');
ylabel('Speech Presence (Binary)');
ylim([-0.1, 1.1]);
grid on;
end

function AGC(config_segment)
    % Loading an audio file (wav) into the variable inputSignal. The sample rate will be saved in the variable sampleRate.
    [inputSignal, sampleRate] = audioread(config_segment('input_path'));

    % Defining gain: the amplification factor  frameSize = config_segment('buffer_size'); % Number of samples per frame
    gain = 10^(config_segment('gain') / 20); % Convert dB gain to linear gain
    
    % Create an array (currently zeroed) for the output, with the same size as the input array (the size of inputSignal)
    outputSignal = zeros(size(inputSignal));
    
    % Divide the audio into parts so that we can work on each part separately. The size of each part will be calculated by dividing the length of the audio by the defined frame size.
    numFrames = floor(length(inputSignal) / config_segment('buffer_size'));
    
    % We will process each part individually
    for i = 1:numFrames
        % Calculate the start and end indices of the audio segment that the loop is currently processing.
        frameStart = (i-1) * config_segment('buffer_size') + 1;
        frameEnd = frameStart + config_segment('buffer_size') - 1;
        % Create a variable currentFrame that will store each segment from start to end (according to the indices we found).
        currentFrame = inputSignal(frameStart:frameEnd);
        
        % Apply the amplification: Multiply the audio segment by the gain control to ensure it is audible and that we do not lose the weaker signals.
        processedFrame = currentFrame * gain;
        
        % Ensure the signals are within the range of -1 to 1. If there is a signal greater than 1, reduce it to 1. If there is a signal less than -1, increase it to -1.
        processedFrame(processedFrame > 1) = 1;
        processedFrame(processedFrame < -1) = -1;
        
        % Place the processed audio into the array we created for the output.
        outputSignal(frameStart:frameEnd) = processedFrame;
    end
    
    % Handling the last segment of the audio if the length of the file is not a multiple of the frame size.
    if frameEnd < length(inputSignal)
        remainingFrame = inputSignal(frameEnd+1:end);
        processedFrame = remainingFrame * gain;
        processedFrame(processedFrame > 1) = 1;
        processedFrame(processedFrame < -1) = -1;
        outputSignal(frameEnd+1:end) = processedFrame;
    end
    
    
    % Save the processed audio in the same folder with a different name, using the same sample rate as the original audio.
    audiowrite(config_segment('output_path'), outputSignal, sampleRate);

    % Create graphs to compare unprocessed and processed audio.
    plotAudioComparison(inputSignal, outputSignal, sampleRate);
end

function plotAudioComparison(inputSignal, outputSignal, sampleRate)
    % Define the time axis for the plot. It starts from 0, goes to the length of inputSignal minus 1, and divides by sampleRate to convert sample indices to time in seconds.
    timeAxis = (0:length(inputSignal)-1) / sampleRate;
    
    % Create a new figure window for the plots: one for the unprocessed audio and one for the processed audio.
    figure;
    
    % Divide the figure into a 2x1 grid and activates the first subplot (top subplot).
    subplot(2, 1, 1);
    plot(timeAxis, inputSignal);
    % Name the graph and the axes
    title('Unprocessed Audio');
    xlabel('Time (s)');
    ylabel('Amplitude');
    
    % Activate the second subplot (bottom subplot) in the 2x1 grid.
    subplot(2, 1, 2);
    plot(timeAxis, outputSignal);
    % Name the graph and the axes
    title('Processed Audio');
    xlabel('Time (s)');
    ylabel('Amplitude');

    % Give the entire figure a title
    sgtitle('Comparison of Unprocessed and Processed Audio');

    % Load the play icon image.
    playIcon = imread('play_icon.jpg');

    % Create a button to play the unprocessed audio.
    uicontrol('Style', 'pushbutton', 'String', playIcon, ...
        'Position', [357 364 20 20], ... % Custom position and size
        'CData', playIcon, ... % Set the play icon
        'Callback', @(src, event) sound(inputSignal, sampleRate));

    % Create a button to play the processed audio.
    uicontrol('Style', 'pushbutton', 'String', playIcon, ...
        'Position', [349 178 20 20], ... % Custom position and size
        'CData', playIcon, ... % Set the play icon
        'Callback', @(src, event) sound(inputSignal, sampleRate));
end

%This code takes a segment of a WAV file performs Shot Time Fourier Transform on the segment
%then it displays the results on a spectrogram. Next it reverse engineers the STFT samples back
%to a WAV file and saves it as: reconstructed_segment2.wav
function STFT(config_segment)
% Read the audio file
[audio, fs] = audioread(config_segment('input_path'));

% Parameters
%segment_start_time = 4; %THIS IS THE STARTING POINT OF THE SAMPLE
%segment_length_time = 2; %THIS IS THE SAMPLE LENGTH
segment_start = fs*config_segment('segment_start_time');  % Starting sample of the segment
segment_length = fs * config_segment('segment_start_time');  % Length of the segment (5 seconds)
window_length = 1024;  % STFT window length
hop_length = 256;  % STFT hop length

% Extract the segment
segment = audio(segment_start:segment_start+segment_length-1);

% Compute STFT
[S, F, T] = stft(segment, fs, 'Window', hann(window_length), 'OverlapLength', window_length - hop_length, 'FFTLength', window_length);

% Plot the spectrogram
figure;
subplot(2,1,1);
imagesc(T, F, mag2db(abs(S)));
axis xy;
colorbar;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram of Audio Segment');

% Plot 3D surface of STFT magnitude
subplot(2,1,2);
surf(T, F, abs(S), 'EdgeColor', 'none');
view(45, 65);  % Adjust the view angle
xlabel('Time (s)');
ylabel('Frequency (Hz)');
zlabel('Magnitude');
title('3D Surface Plot of STFT Magnitude');
colorbar;

% Reconstruct the signal using IFFT
reconstructed_segment = istft(S, fs, 'Window', hann(window_length), 'OverlapLength', window_length - hop_length, 'FFTLength', window_length);

% Write the reconstructed segment to a new WAV file
audiowrite(config_segment('output_path'), reconstructed_segment, fs);

% Plot original and reconstructed signals
figure;
subplot(2,1,1);
plot((0:length(segment)-1)/fs, segment);
title('Original Segment');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2,1,2);
plot((0:length(reconstructed_segment)-1)/fs, reconstructed_segment);
title('Reconstructed Segment');
xlabel('Time (s)');
ylabel('Amplitude');
end

function noise_reduction(config_segment)

    % Read the input audio file
    [audio, sample_rate] = audioread(config_segment('input_path'));
    audio = audio / max(abs(audio));  % Normalize audio
    
    % Compute the spectrogram
    [S, F, T] = custom_spectrogram(audio, config_segment('fft_size'), config_segment('hop_length'), sample_rate);
    
    % Estimate noise profile using low percentile
    noise_profile = prctile(abs(S).^2, 20, 2);
    
    % Wiener filter
    SNR = max((abs(S).^2 ./ (noise_profile + eps)) - 1, 0);
    gain = SNR ./ (SNR + 1);
    S_reduced = S .* gain;
    
    % Inverse transform
    audio_reduced = custom_istft(S_reduced, config_segment('hop_length'));
    audio_reduced = audio_reduced / max(abs(audio_reduced));
    
    % Save the filtered audio to the output file
    audiowrite(config_segment('output_path'), audio_reduced, sample_rate);
    disp(['Filtered WAV file created successfully: ' config_segment('output_path')]);
    
    % Plot waveforms and spectrograms
    plot_waveforms_and_spectrograms(config_segment('input_path'), config_segment('output_path'));
end

function plot_waveforms_and_spectrograms(original_file, filtered_file)
    [original, sample_rate] = audioread(original_file);
    filtered = audioread(filtered_file);
    
    time_original = (0:length(original)-1) / sample_rate;
    time_filtered = (0:length(filtered)-1) / sample_rate;
    
    figure;
    
    % Original waveform
    subplot(2, 2, 1);
    plot(time_original, original);
    title('Original Waveform');
    xlabel('Time (seconds)');
    ylabel('Amplitude');
    
    % Filtered waveform
    subplot(2, 2, 2);
    plot(time_filtered, filtered);
    title('Filtered Waveform');
    xlabel('Time (seconds)');
    ylabel('Amplitude');
    
    % Original spectrogram
    subplot(2, 2, 3);
    [S, F, T] = custom_spectrogram(original, 1024, 512, sample_rate);
    imagesc(T, F, 20*log10(abs(S))); axis xy;
    title('Original Spectrogram');
    xlabel('Time (seconds)');
    ylabel('Frequency (Hz)');
    
    % Filtered spectrogram
    subplot(2, 2, 4);
    [S, F, T] = custom_spectrogram(filtered, 1024, 512, sample_rate);
    imagesc(T, F, 20*log10(abs(S))); axis xy;
    title('Filtered Spectrogram');
    xlabel('Time (seconds)');
    ylabel('Frequency (Hz)');
end

function [S, F, T] = custom_spectrogram(x, window_size, hop_length, fs)
    % Compute spectrogram without using Signal Processing Toolbox
    num_samples = length(x);
    num_windows = floor((num_samples - window_size) / hop_length) + 1;
    S = zeros(window_size, num_windows);
    
    for i = 1:num_windows
        start_idx = (i-1) * hop_length + 1;
        end_idx = start_idx + window_size - 1;
        window = x(start_idx:end_idx);
        S(:, i) = fft(window);
    end
    
    S = S(1:window_size/2+1, :);  % Keep only positive frequencies
    F = linspace(0, fs/2, size(S, 1));
    T = linspace(0, num_samples/fs, num_windows);
end

function x_reconstructed = custom_istft(S, hop_length)
    % Inverse transform without using Signal Processing Toolbox
    [num_freq, num_frames] = size(S);
    window_size = (num_freq - 1) * 2;
    
    x_reconstructed = zeros((num_frames-1) * hop_length + window_size, 1);
    
    for i = 1:num_frames
        start_idx = (i-1) * hop_length + 1;
        end_idx = start_idx + window_size - 1;
        
        frame = ifft([S(:, i); conj(flipud(S(2:end-1, i)))], 'symmetric');
        x_reconstructed(start_idx:end_idx) = x_reconstructed(start_idx:end_idx) + real(frame);
    end
end

function pitch_estimation(config_segment)

system(['python "inputs\pitch estimation.py" "' config_segment('input_path') '" "' config_segment('output_path') '"']);
end

function Audio_Speed_Modification(config_segment)
%Change the speed of the .WAV file in the input file path by config_segment('speed_factor') and save the result to config_segment('output_path')

%The code doesn't work if config_segment('speed_factor')>2, and of course it doesn't work if config_segment('speed_factor')<0
if config_segment('speed_factor')>2 || config_segment('speed_factor')<=0
    %msg="The given speed factor of %s is either bigger than 2 or lower than 0", config_segment('speed_factor');
    error("The given speed factor of %d is either bigger than 2 or lower/equal to 0." + ...
        "\nThe speed factor must be between 0 and 2", config_segment('speed_factor'));
end

%Read the audio file
[input_audio, sample_rate] = audioread(config_segment('input_path'));

%If config_segment('speed_factor') = 1, no changes are to be made - write the input file to the save address
if config_segment('speed_factor')==1
    copyfile(config_segment('input_path'), config_segment('output_path'));
    return
end

%Parameters
win_size = 1024;

%Set the overlap size/length
overlap = round(win_size - config_segment('speed_factor')*win_size/2);

%Create the hann Window
test = linspace(0, win_size, win_size);
hann_window = sin(pi*test/win_size).^2';

%Create input audio buffer with the overlap
buffered_audio = buffer(input_audio, win_size, overlap, 'nodelay');

%Create a matrix of the hann windows
windows_arr = repmat(hann_window, 1, size(buffered_audio, 2));

%Pass the hash windows on the input audio
windowed_audio = buffered_audio .* windows_arr;

%Create the output sound vector, which we will restore the sound to
output = zeros(((size(windowed_audio,2)-1) * win_size/2)+win_size, 1);

%Restore the modified sound (sped up/down)
for i=1:size(windowed_audio, 2)
    seg_start = (i-1)*win_size/2+1; %Set writing current start location
    seg_end = win_size/2*(i+1); %Set writing current end location
    output(seg_start:seg_end)  = output(seg_start:seg_end) + windowed_audio(:,i); %Update the output sound vector
end

%Normalize the output sound vector
output = output/max(abs(output));
config = toml.read('config.toml');
%Save the modified sound file to the parameter address

% old
audiowrite(append(config_segment('output_path'), "\modified_audio.wav"), output, sample_rate); 
% new
audiowrite(conf_segment('output_path'), output, sample_rate);

end

function decimation_antialiasing(config_segment)

    % Set default decimation factor if not provided
decimation_factor=config_segment('decimation_factor');
% if nargin < 2
% decimation_factor = 4;
% end

% Getting the array of the file and the freq of the sampling.
% y - array, Fs - frequency
[y, Fs] = audioread(config_segment('input_path'));
if size(y, 2) > 1
    y= mean(y, 2);
end    
% Define the new freq
Fs_new = round(Fs / decimation_factor);
T=1/Fs_new;

% Create the anti-aliasing filter (lowpass filter)
filter_length = 2001;  % Odd number for symmetric filter
t_sinc = (-(filter_length-1)/2 : (filter_length-1)/2)*T;
S1 = sinc(t_sinc/T);

% Apply the anti-aliasing filter
y_filtered = conv(y, S1, 'same');

% Perform decimation
y_deci = y_filtered(1:decimation_factor:end);
% Write the signal to a .wav file
    audiowrite(config_segment('output_path'), y_deci, Fs_new);
    
    disp(['Audio file saved at ', config_segment('output_path')]);
if config_segment('FOH')
    FOH_func(config_segment)
elseif config_segment('SHANON')
        SHANON_func(config_segment)
else 
    ZOH_func(config_segment)
end
end

function FOH_func(config_segment)
% Getting the array of the file and the freq of the sampling.
% y - array, Fs - frequency
[y, Fs] = audioread(config_segment('output_decimation_path'));

interpolation_factor=config_segment('interpolation_factor');
% Set default interpolatin factor if not provided.
% if nargin < 2
% interpolation_factor = 4;
% end

% Setting new frequency.
Fs_new = Fs*interpolation_factor;
    % Length of the interpolatin array
    n = length(y);
    
    % Each element in the original array will have interpolaion factor zeros after it,
    % except the last element which has no zeros following it.
    y_interp_length = n + (interpolation_factor-1) * (n - 1);

    % Initialize the interpolation array with zeros
    y_interp = zeros(1, y_interp_length);

    % Insert the elements of the input array and zeros into the output array
    % Perform interploation.
    y_interp(1:interpolation_factor:end) = y;

% Setting T for the FOH filter.
T=1/Fs_new;
% Time vector for the function.
t = linspace(-T, T, 2*interpolation_factor+1);
% Creating the ZOH filer.
w = (1 - abs(t)/T) .* (abs(t) < T);
y_FOH = conv(y_interp, w, 'same');
% Write the signal to a .wav file
    audiowrite(config_segment('output_interpolation_path'), y_FOH, Fs_new);
    
    disp(['Audio file saved at ', config_segment('output_interpolation_path')]);
end

function SHANON_func(config_segment)

% Getting the array of the file and the freq of the sampling.
% y - array, Fs - frequency
[y, Fs] = audioread(config_segment('output_decimation_path'));

interpolation_factor=config_segment('interpolation_factor');
% Set default interpolatin factor if not provided.
% if nargin < 2
% interpolation_factor = 4;
% end

% Setting new frequency.
Fs_new = Fs*interpolation_factor;
    % Length of the interpolatin array
    n = length(y);
    
    % Each element in the original array will have interpolaion factor zeros after it,
    % except the last element which has no zeros following it.
    y_interp_length = n + (interpolation_factor-1) * (n - 1);

    % Initialize the interpolation array with zeros.
    y_interp = zeros(1, y_interp_length);

    % Insert the elements of the input array and zeros into the output array
    % Perform interploation.
    y_interp(1:interpolation_factor:end) = y;

% Setting T for the SHANON filter.
T=1/Fs_new;
filter_length = 2001;  % Odd number for symmetric filter.
% Time vector for the sinc function.
t_sinc = (-(filter_length-1)/2 : (filter_length-1)/2)/Fs;
%creating the SHANON filter.
S1 = sinc(t_sinc/T);

% Doing the convolution to apply the filter
y_SHANON = conv(y_interp, S1, 'same');

% Write the signal to a .wav file
    audiowrite(config_segment('output_interpolation_path'), y_SHANON, Fs_new);
    
    disp(['Audio file saved at ', config_segment('output_interpolation_path')]);
end

function ZOH_func(config_segment)
% Getting the array of the file and the freq of the sampling.
% y - array, Fs - frequency
[y, Fs] = audioread(config_segment('output_decimation_path'));

interpolation_factor = config_segment('interpolation_factor');
% Set default interpolatin factor if not provided.
% if nargin < 2
% interpolation_factor = 4;
% end

% Setting new frequency.
Fs_new = Fs*interpolation_factor;
    % Length of the interpolatin array
    n = length(y);
    
    % Each element in the original array will have interpolaion factor zeros after it,
    % except the last element which has no zeros following it.
    y_interp_length = n + (interpolation_factor-1) * (n - 1);

    % Initialize the interpolation array with zeros
    y_interp = zeros(1, y_interp_length);

    % Insert the elements of the input array and zeros into the output array
    % Perform interploation.
    y_interp(1:interpolation_factor:end) = y;

% Setting T for the ZOH filter. 
T=1/Fs_new;
% Time vector for the function.
t = linspace(-T/2, T*1.5, 2*interpolation_factor+1);
% Creating the ZOH filter.
w = double((t >= 0) & (t < T));
y_ZOH = conv(y_interp, w, 'same');
% Write the signal to a .wav file
    audiowrite(config_segment('output_interpolation_path'), y_ZOH, Fs_new);
    
    disp(['Audio file saved at ', config_segment('output_interpolation_path')]);
end
    
function SSB(config_segment)
% Read audio file
[audio, mfs] = audioread(config_segment('input_path')); %%%%%%%%%%%%%%%%% here you write the name of the file with the .wav ending
if size(audio, 2) > 1
    audio = audio(:, 1); % Take left channel if stereo
end

% Calculate the duration of the audio file
tend = length(audio) / mfs;

% Create time vector for original signal
t = (0:length(audio)-1)' / mfs;

% Resample the audio signal in two steps if necessary
ns = round(tend * config_segment('fs'));
if ns > 2^30 || length(audio) > 2^30 % Check if we need to use an intermediate step
    % First, resample to an intermediate rate
    intermediate_fs = min(config_segment('fs'), mfs) * 2; % Choose an appropriate intermediate rate
    [m_intermediate, ~] = resample(audio, intermediate_fs, mfs);
    
    % Then, resample to the final rate
    [m, t_resampled] = resample(m_intermediate, config_segment('fs'), intermediate_fs);
else
    % If not necessary, resample directly
    [m, t_resampled] = resample(audio, ns, length(audio));
end

t_resampled = (0:length(m)-1)' / config_segment('fs');

% Normalize the audio signal
m = m / 32768; % Assuming 16-bit audio

% Frequency vector for single-sided spectral plotting
f = linspace(0, config_segment('fs')/2, floor(length(m)/2));

% Carrier signal
xc = cos(2 * pi * config_segment('fc') * t_resampled);

% Double sideband suppressed carrier modulation (DSBSC)
sm = m .* xc;
Sm = fftshift(fft(sm));

% Single sideband modulation (SSB)
analytic_signal = hilbert(m);
ssb_sm = real(analytic_signal .* exp(1i * 2 * pi * config_segment('fc') * t_resampled));
Ssb_Sm = fftshift(fft(ssb_sm));

% Demodulation
rxerr = 0; % Deliberate error in receive carrier frequency
rm = sm .* cos(2 * pi * (config_segment('fc') + rxerr) * t_resampled);
[b, a] = butter(7, 0.2);
rlpf = filter(b, a, rm);
Rlpf = fftshift(fft(rlpf));

% Save Output
audiowrite(config_segment('output_path'), t_resampled, ssb_sm);

% Plot Original Signal
figure;
plot(t_resampled, m);
title('Original Signal');
xlabel('Time [sec]');
ylabel('Amplitude');
grid on;



% Plot Demodulated Signal
figure;
plot(t_resampled, rlpf);
xlabel('Time [sec]');
title('Demodulated Signal');
ylabel('Amplitude');
grid on;

% Plot SSB Modulated Signal
figure;
plot(t_resampled, ssb_sm);
xlabel('Time [sec]');
title('SSB Modulated Signal');
ylabel('Amplitude');
grid on;
end
