import librosa
import librosa.display
import numpy as np
import soundfile as sf
import random
import io
import os
import matplotlib.pyplot as plt


SPECTROGRAM_DIR = "spectrograms/"
os.makedirs(SPECTROGRAM_DIR, exist_ok=True)  


import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOUND_FILES = {
    "walking": os.path.join(BASE_DIR, "sounds/walking.wav"),
    "running": os.path.join(BASE_DIR, "sounds/running.wav"),
    "wind": os.path.join(BASE_DIR, "sounds/wind.wav"),
    "explosion": os.path.join(BASE_DIR, "sounds/explosion.wav")
}



def save_spectrogram(audio_data, sample_rate, event_type):

    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128, fmax=8000)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max) 



    filename = event_type + "_" + str(random.randint(1000, 9999)) + ".png"

    file_path = os.path.join(SPECTROGRAM_DIR, filename)


    plt.figure(figsize=(5, 5))
    librosa.display.specshow(mel_spectrogram_db, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.axis('off')  
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close() 

    print(" Spectrogram saved:", file_path)
    return file_path


def modify_sound(event_type):



    sound_path = SOUND_FILES.get(event_type)
    if not sound_path:

        return None  


    audio_data, sample_rate = librosa.load(sound_path, sr=None)  


    
    
    speed_factor = random.uniform(0.8, 1.2)
    audio_data = librosa.effects.time_stretch(audio_data, rate=speed_factor)

 
    pitch_factor = random.uniform(-3, 3)
    audio_data = librosa.effects.pitch_shift(audio_data, sr=sample_rate, n_steps=pitch_factor)


    volume_factor = random.uniform(0.7, 1.3)
    audio_data *= volume_factor  


    spectrogram_path = save_spectrogram(audio_data, sample_rate, event_type)

    print(" Sound processed and spectrogram saved for:", event_type)
    return spectrogram_path  



def modify_sound_with_distance(event_type, distance):

    pitch_shift = random.uniform(-2, 2) * (distance / 20)
    speed_factor = max(0.5, random.uniform(0.8, 1.2) * (1 - distance / 40))  
    volume_factor = max(0.1, 1 - distance / 30)


    audio_data, sample_rate = librosa.load(SOUND_FILES[event_type], sr=None)
    audio_data = librosa.effects.pitch_shift(audio_data, sr=sample_rate, n_steps=pitch_shift)
    audio_data = librosa.effects.time_stretch(audio_data, rate=speed_factor)
    audio_data *= volume_factor

    return save_spectrogram(audio_data, sample_rate, event_type)
