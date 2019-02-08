from collections import defaultdict
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

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


# Printing
zero_data = np.zeros(len(data))
for zero in zero_crossings:
    zero_data[zero] = 30000

length = 30000
start = 30000
plt.plot(filtered_data[start : start+length])
plt.plot(zero_data[start : start+length])
plt.ylim(-30000, 30000)
plt.grid(True)
plt.show()


# Split to frames
frames = list()
for i in range(0, len(zero_crossings)-1):
    frames.append(data[zero_crossings[i]:zero_crossings[i+1]])
print("len(frames):", len(frames))
