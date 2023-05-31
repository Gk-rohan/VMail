import whisper
import speech_recognition as sr
# from playsound import playsound
import tempfile
import io
import os
from pydub import AudioSegment

r = sr.Recognizer()

temp_dir = tempfile.mkdtemp()
save_path = os.path.join(temp_dir, "temp.wav")

model = whisper.load_model('base')
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source, duration=1)
    # playsound('speak.mp3')
    print('speak')
    audio = r.listen(source, phrase_time_limit=10)
    # response = r.recognize_google(audio)
    data = io.BytesIO(audio.get_wav_data())
    audio_clip = AudioSegment.from_file(data)
    audio_clip.export(save_path, format="wav")
    response = model.transcribe(save_path)

print(response["text"])

# import whisper
# model = whisper.load_model('base')
# text = model.transcribe("Recording (4).m4a")

# #printing the transcribe
# print(text['text'])

# from faster_whisper import WhisperModel

# model_size = "small"

# # or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

# segments, info = model.transcribe("audio.mp3", beam_size=5)

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))