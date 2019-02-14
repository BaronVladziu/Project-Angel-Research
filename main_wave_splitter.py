from collections import defaultdict
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import python_speech_features as psf


### CREATE TRANSCRIPTION DICTIONARY ###
transcriptions = defaultdict(list)

text_file = open("./transcriptions/transcriptions.txt", "r")
lines = text_file.read().split("\n")
text_file.close()
for line in lines:
    words = line.split(" ")
    transcriptions[words[0]] = words[2:]
    # print(words[0], " - ", transcriptions[words[0]])


### TRANSCRIBE RECORDINGS ###
file_name = "rec2"
text_file = open("./wavs/" + file_name + ".txt", "r")
lines = text_file.read().split("\n")
text_file.close()
print(lines)

phones = list()
phones.append("PAUSE")
for line in lines:
    for word in line.split(" "):
        for phone in transcriptions[word.lower()]:
            phones.append(phone)
        phones.append("PAUSE")
print(phones)


### SPLIT WAVE TO FRAMES ###
min_voice_frequency = 73
max_voice_frequency = 1000
fs, data = wavfile.read("./wavs/" + file_name + ".wav")
print("Wave length:", len(data))

min_voice_period = int(fs/max_voice_frequency)
max_voice_period = int(fs/min_voice_frequency) + 1
print("Voice period: <", min_voice_period, " - ", max_voice_period, ">")

# Filtering
b, a = signal.butter(10, max_voice_frequency/fs, "low")
filtered_data = signal.filtfilt(b, a, data)

# Calculating derivative
dt = 1/fs
der = np.zeros(len(data))
der[0] = (filtered_data[1] - filtered_data[0])/dt
for i in range(1, len(data) - 1):
    der[i] = (filtered_data[i+1] - filtered_data[i-1])/(2*dt)
der[len(data) - 1] = (filtered_data[len(data) - 1] - filtered_data[len(data) - 2])/dt

# Finding local maximas
zero_crossings = list()
for i in range(0, len(data) - 1):
    if der[i] > 0 and der[i+1] < 0:
        zero_crossings.append(i)

# Removing unnecessary maximas
global_threshold = 0.1
i = 0
while i < len(zero_crossings) - 1:
    # Calculate period
    subdata_length = min(zero_crossings[i]+2*max_voice_period-1, len(data)-1) - zero_crossings[i]
    window_length = subdata_length - max_voice_period
    diff = np.zeros(max_voice_period + 1)
    cum_diff = np.zeros(max_voice_period + 1)
    diff[0] = 1
    cum_diff[0] = 1
    act_min_id = 0
    act_min_val = 1
    found_global_th = False
    for j in range(1, max_voice_period+1):
        d = filtered_data[zero_crossings[i] : zero_crossings[i]+window_length] - filtered_data[zero_crossings[i]+j : zero_crossings[i]+j+window_length]
        diff[j] = np.sum(np.power(d, 2))
        cum_diff[j] = diff[j] / np.mean(diff[0:j+1])
        if found_global_th and cum_diff[j] > cum_diff[j-1]:
            act_min_id = j-1
            break
        else:
            if cum_diff[j] < global_threshold:
                found_global_th = True
            elif cum_diff[j] < act_min_val:
                act_min_id = j
                act_min_val = cum_diff[j]
    period = act_min_id
    # Remove unnecessary maximas
    while i+2 < len(zero_crossings) and abs(zero_crossings[i+2] - zero_crossings[i] - period) < abs(zero_crossings[i+1] - zero_crossings[i] - period):
        zero_crossings.pop(i+1)
    i += 1

# Split to frames
frames = list()
for i in range(0, len(zero_crossings)-1):
    frames.append(data[zero_crossings[i]:zero_crossings[i+1]])
print("len(frames):", len(frames))

# # Printing
# zero_data = np.zeros(len(data))
# for zero in zero_crossings:
#     zero_data[zero] = 30000
#
# length = 30000
# start = 30000
# plt.plot(filtered_data[start : start+length])
# plt.plot(zero_data[start : start+length])
# plt.ylim(-30000, 30000)
# plt.grid(True)
# plt.show()


### PHONE MODELS ###
class PhoneBegining:
    name = ""
    id = 0

    def __init__(self, name, id):
        self.name = name
        self.id = id

    def __str__(self):
        return "( " + str(self.name) + ", " + str(self.id) + " )"

    def __repr__(self):
        return self.__str__()


class PhoneModel:
    def __init__(self, name, length):
        self.name = name
        self.length = length
        self.calculated_freqs = np.zeros(self.length)
        self.calculated_consts = np.zeros(self.length)
        self.frames = list()
        self.freqs = list()

    def draw(self):
        plt.plot(self.calculated_freqs*110 + self.calculated_consts)
        plt.plot(self.calculated_freqs*220 + self.calculated_consts)
        plt.plot(self.calculated_freqs*440 + self.calculated_consts)
        plt.plot(self.calculated_freqs*880 + self.calculated_consts)
        plt.plot(self.calculated_freqs*1760 + self.calculated_consts)
        plt.grid(True)
        plt.show()

    def add(self, frame, freq):
        self.frames.append(signal.resample(frame, self.length))
        self.freqs.append(freq)

    def calculate(self):
        sum_val = np.zeros(self.length)
        sum_valfreq = np.zeros(self.length)
        sum_freq = 0
        sum_freq2 = 0
        for i in range(0, len(self.frames)):
            sum_val += self.frames[i]
            sum_valfreq += self.frames[i] * self.freqs[i]
            sum_freq += self.freqs[i]
            sum_freq2 += np.power(self.freqs[i], 2)
        self.calculated_freqs = (sum_valfreq * len(self.frames) - sum_val * sum_freq) / (len(self.frames) * sum_freq2 - np.power(sum_freq, 2))
        self.calculated_consts = (sum_val * (sum_freq2 - np.power(sum_freq, 2) + sum_freq) - sum_valfreq * len(self.frames)) / (len(self.frames) * (sum_freq2 - np.power(sum_freq, 2)))

    def count_frame_to_model_difference(self, frame, freq):
        resampled_frame = signal.resample(frame, self.length)
        model_frame = self.calculated_freqs*freq + self.calculated_consts
        return np.sum(np.power(resampled_frame - model_frame, 2))

    def get_frame(self, freq, fs):
        return signal.resample(self.calculated_freqs*freq + self.calculated_consts, int(fs/freq))


def recalculate_models(phone_models, phone_borders, frames, fs):
    for name in phone_models:
        phone_models[name] = PhoneModel(name, 2048)
    for i in range(0, len(phone_borders) - 1):
        if phone_borders[i].name != "PAUSE":
            for j in range(phone_borders[i].id, phone_borders[i + 1].id):
                phone_models[phone_borders[i].name].add(frames[j], fs / len(frames[j]))
    for j in range(phone_borders[-1].id, len(frames)):
        if phone_borders[-1].name != "PAUSE":
            phone_models[phone_borders[-1].name].add(frames[j], fs / len(frames[j]))

    phone_models["PAUSE"].add(np.zeros(2048), 100)
    phone_models["PAUSE"].add(np.zeros(2048), 1000)

    for name in phone_models:
        phone_models[name].calculate()
    # print(phone_borders)


# ### CREATE INITIAL PHONE BORDERS (Iterative border shifting) ###
#
#
# phone_borders = list()
# init_phone_length = int(len(frames)/len(phones))
# for i in range(0, len(phones)):
#     phone_borders.append(PhoneBegining(phones[i], i*init_phone_length))
#
# # Create phone models
# phone_models = dict()
# for name in phones:
#     phone_models[name] = PhoneModel(name, 2048)
# recalculate_models(phone_models, phone_borders, frames, fs)
#
#
# ### FIT MODELS TO PHONES (Iterative border shifting) ###
# def where_to_move_border(left_frame, left_freq, right_frame, right_freq, left_model, right_model):
#     left_to_left_difference = left_model.count_frame_to_model_difference(left_frame, left_freq)
#     left_to_right_difference = right_model.count_frame_to_model_difference(left_frame, left_freq)
#     right_to_left_difference = left_model.count_frame_to_model_difference(right_frame, right_freq)
#     right_to_right_difference = right_model.count_frame_to_model_difference(right_frame, right_freq)
#     if left_to_left_difference > left_to_right_difference and right_to_left_difference > right_to_right_difference:
#         return "left"
#     elif left_to_left_difference < left_to_right_difference and right_to_left_difference < right_to_right_difference:
#         return "right"
#     else:
#         return "stay"
#
#
# for j in range(0, 100):
#     for i in range(1, len(phone_borders)):
#         left_frame = frames[phone_borders[i].id-1]
#         left_freq = fs / len(frames[phone_borders[i].id-1])
#         right_frame = frames[phone_borders[i].id]
#         right_freq = fs / len(frames[phone_borders[i].id])
#         left_model = phone_models[phone_borders[i-1].name]
#         right_model = phone_models[phone_borders[i].name]
#         move_decision = where_to_move_border(left_frame, left_freq, right_frame, right_freq, left_model, right_model)
#         if move_decision == "left":
#             if phone_borders[i-1].id + 2 < phone_borders[i].id:
#                 phone_borders[i].id -= 1
#         elif move_decision == "right":
#             if i == len(frames) - 1:
#                 if phone_borders[i].id < len(frames) - 1:
#                     phone_borders[i].id += 1
#             else:
#                 if phone_borders[i].id < phone_borders[i+1].id - 2:
#                     phone_borders[i].id += 1
#     recalculate_models(phone_models, phone_borders, frames, fs)


### MFCC CALCULATION ###
class MFCC:
    def __init__(self, signal, fs):
        self.coeffs = psf.mfcc(signal, fs, nfft=2048)

    def calc_difference(self, other):
        return np.sum(np.power(self.coeffs - other.coeffs, 2))


frame_mfccs = list()
for frame in frames:
    frame_mfccs.append(MFCC(frame, fs))


### BETTER PHONEME SEGMENTATION ###
# frame_diffs = np.zeros(len(frames) - 1)
# for i in range(0, len(frame_mfccs) - 1):
#     frame_diffs[i] = frame_mfccs[i].calc_difference(frame_mfccs[i+1])
# border_inds = np.sort(np.argpartition(frame_diffs, -len(phones))[-len(phones):])
# print(border_inds)


### OPTIMAL PHONEME SEGMENTATION (Square error) ###
class Segment:
    def __init__(self, start_id, end_id, frame_mfccs):
        self.start_id = start_id
        self.end_id = end_id
        self.__calc_mean_and_sum(frame_mfccs)

    def __calc_mean_and_sum(self, frame_mfccs):
        self.mean = np.zeros(np.shape(frame_mfccs[0].coeffs))
        for i in range(self.start_id, self.end_id + 1):
            self.mean += frame_mfccs[i].coeffs
        self.mean /= self.end_id + 1 - self.start_id
        sum = np.zeros(np.shape(frame_mfccs[0].coeffs))
        for i in range(self.start_id, self.end_id + 1):
            sum += np.power(frame_mfccs[i].coeffs - self.mean, 2)
        self.value = np.sum(sum)

    def merge_with(self, other, frame_mfccs):
        self.start_id = min(self.start_id, other.start_id)
        self.end_id = max(self.end_id, other.end_id)
        self.__calc_mean_and_sum(frame_mfccs)

    def get_fit_value(self):
        return self.value


segments = list()
for i in range(0, len(frames)):
    segments.append(Segment(i, i, frame_mfccs))
while len(segments) > len(phones):
    merge_segments = list()
    for i in range(0, len(segments) - 1):
        merge_segments.append(Segment(segments[i].start_id, segments[i+1].end_id, frame_mfccs))
    merge_costs = np.zeros(len(segments) - 1)
    for i in range(0, len(segments) - 1):
        merge_costs[i] = merge_segments[i].value - segments[i].value - segments[i+1].value
    left = np.argmin(merge_costs)
    segments.pop(left)
    segments.pop(left)
    segments.insert(left, merge_segments[left])

border_inds = list()
for segment in segments:
    border_inds.append(segment.start_id)


# Draw phone borders
length = 1000000
start = 0
border_array = np.zeros(len(data))
for border in border_inds:
    border_array[zero_crossings[border]] = 30000
plt.plot(data[start:start+length])
plt.plot(border_array[start:start+length])
plt.ylim(-30000, 30000)
plt.grid(True)
plt.show()


# ### CREATE SENTHESIS REQUEST ###
# class SynthPhone:
#     def __init__(self, phone, duration, pitch):
#         self.phone = phone
#         self.duration = duration
#         self.pitch = pitch
#
#
# synth_phones = list()
# synth_phones.append(SynthPhone("PAUSE", 80, 220))
# synth_phones.append(SynthPhone("f", 20, 220))
# synth_phones.append(SynthPhone("ai", 180, 220))
# synth_phones.append(SynthPhone("PAUSE", 20, 220))
# synth_phones.append(SynthPhone("E", 80, 220))
# synth_phones.append(SynthPhone("n", 20, 220))
# synth_phones.append(SynthPhone("A", 160, 220))
# synth_phones.append(SynthPhone("f", 20, 220))
# synth_phones.append(SynthPhone("PAUSE", 20, 220))
# synth_phones.append(SynthPhone("E", 80, 220))
# synth_phones.append(SynthPhone("w", 20, 220))
# synth_phones.append(SynthPhone("e", 180, 220))
# synth_phones.append(SynthPhone("I", 20, 220))
# synth_phones.append(SynthPhone("PAUSE", 100, 220))
#
#
# ### SYNTHESIZE SENTENCE ###
# synthesized_signal = np.zeros(0)
# for synth_phone in synth_phones:
#     for i in range(0, synth_phone.duration):
#         frame = phone_models[synth_phone.phone].get_frame(synth_phone.pitch, fs)
#         if synth_phone.phone != "PAUSE":
#             frame /= np.max(np.abs(frame))
#         synthesized_signal = np.concatenate([synthesized_signal, frame])
# synthesized_signal /= np.max(np.abs(synthesized_signal))
#
#
# wavfile.write('test.wav', fs, synthesized_signal)
# sd.play(synthesized_signal, fs)
# plt.plot(synthesized_signal)
# plt.grid(True)
# plt.show()
#
#
