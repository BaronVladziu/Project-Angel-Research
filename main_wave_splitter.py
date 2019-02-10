from collections import defaultdict
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd


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


### CREATE INITIAL PHONE BORDERS ###
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


phone_borders = list()
init_phone_length = int(len(frames)/len(phones))
for i in range(0, len(phones)):
    phone_borders.append(PhoneBegining(phones[i], i*init_phone_length))
print(phone_borders)

# Create phone models
phone_models = dict()
for name in phones:
    phone_models[name] = PhoneModel(name, 2048)

for i in range(0, len(phone_borders) - 1):
    if phone_borders[i].name != "PAUSE":
        for j in range(phone_borders[i].id, phone_borders[i+1].id):
            phone_models[phone_borders[i].name].add(frames[j], fs/len(frames[j]))
for j in range(phone_borders[-1].id, len(frames)):
    if phone_borders[-1].name != "PAUSE":
        phone_models[phone_borders[-1].name].add(frames[j], fs/len(frames[j]))

phone_models["PAUSE"].add(np.zeros(2048), 100)
phone_models["PAUSE"].add(np.zeros(2048), 1000)

for name in phone_models:
    phone_models[name].calculate()


### FIT MODELS TO PHONES ###
def where_to_move_border(left_frame, left_freq, right_frame, right_freq, left_model, right_model):
    left_to_left_difference = left_model.count_frame_to_model_difference(left_frame, left_freq)
    left_to_right_difference = right_model.count_frame_to_model_difference(left_frame, left_freq)
    right_to_left_difference = left_model.count_frame_to_model_difference(right_frame, right_freq)
    right_to_right_difference = right_model.count_frame_to_model_difference(right_frame, right_freq)
    if left_to_left_difference > left_to_right_difference and right_to_left_difference > right_to_right_difference:
        return "left"
    elif left_to_left_difference < left_to_right_difference and right_to_left_difference < right_to_right_difference:
        return "right"
    else:
        return "stay"

# TODO
# for i in range(1, len(phone_borders)):
#


### CREATE SENTHESIS REQUEST ###
class SynthPhone:
    def __init__(self, phone, duration, pitch):
        self.phone = phone
        self.duration = duration
        self.pitch = pitch


synth_phones = list()
synth_phones.append(SynthPhone("PAUSE", 80, 220))
synth_phones.append(SynthPhone("f", 20, 220))
synth_phones.append(SynthPhone("ai", 180, 220))
synth_phones.append(SynthPhone("PAUSE", 20, 220))
synth_phones.append(SynthPhone("E", 80, 220))
synth_phones.append(SynthPhone("n", 20, 220))
synth_phones.append(SynthPhone("A", 160, 220))
synth_phones.append(SynthPhone("f", 20, 220))
synth_phones.append(SynthPhone("PAUSE", 20, 220))
synth_phones.append(SynthPhone("E", 80, 220))
synth_phones.append(SynthPhone("w", 20, 220))
synth_phones.append(SynthPhone("e", 180, 220))
synth_phones.append(SynthPhone("I", 20, 220))
synth_phones.append(SynthPhone("PAUSE", 100, 220))


### SYNTHESIZE SENTENCE ###
synthesized_signal = np.zeros(0)
for synth_phone in synth_phones:
    for i in range(0, synth_phone.duration):
        frame = phone_models[synth_phone.phone].get_frame(synth_phone.pitch, fs)
        if synth_phone.phone != "PAUSE":
            frame /= np.max(np.abs(frame))
        synthesized_signal = np.concatenate([synthesized_signal, frame])
synthesized_signal /= np.max(np.abs(synthesized_signal))


wavfile.write('test.wav', fs, synthesized_signal)
sd.play(synthesized_signal, fs)
plt.plot(synthesized_signal)
plt.grid(True)
plt.show()


