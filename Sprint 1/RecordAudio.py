import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
 
# Sampling frequency
freq = 44100
  
# Recording duration
duration = 5
  
# Start recorder with the given values 
# of duration and sample frequency
recording = sd.rec(int(duration * freq), 
                   samplerate=freq, channels=2)

print("start recording")  
# Record audio for the given number of seconds
sd.wait()
print("end recording")

  
# 2 ways of saving:
write("recording0.wav", freq, recording)
wv.write("recording1.wav", recording, freq, sampwidth=2)