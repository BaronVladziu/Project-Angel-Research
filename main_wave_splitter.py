from collections import defaultdict
from scipy.io import wavfile
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

zero_crossings = list()
for i in range(0, len(data) - 2):
    if data[i] < 0 and data[i+1] > 0:
        zero_crossings.append(i)

for i in range(0, len(zero_crossings) - 1):
    # Calculate period
    correlation = np.zeros(max_voice_period)
    for j in range(min_voice_period, max_voice_period - 1):
        for k in range(zero_crossings[i], zero_crossings[i] + max_voice_period - 1):
            correlation[j - min_voice_period] += data[k]*data[k+j]
    period = np.argmax(correlation)
    # Remove unnecessary zeros





zero_data = np.zeros(len(data))
for zero in zero_crossings:
    zero_data[zero] = 30000

length = 5000
start = 50000
plt.plot(data[start : start+length])
plt.plot(zero_data[start : start+length])
# plt.plot(peaks[start : start+length])
plt.ylim(-30000, 30000)
plt.grid(True)
plt.show()
