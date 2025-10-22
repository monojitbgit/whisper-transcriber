import time
import torch
import whisper
import os

torch.set_num_threads(torch.get_num_threads())  # Use all CPU cores

audio_file = "audio/Zoom 13.06.2023.mp3"

model = whisper.load_model("medium")  # More accurate than "tiny" or "base"

start_time = time.time()
result = model.transcribe(audio_file, fp16=False, language="english",  task="transcribe")  # Force English
end_time = time.time()

# Print results
print("Transcription:", result["text"])
print(f"⏳ Time taken: {end_time - start_time:.2f} seconds")

# Generate output filename with .txt extension
base_name = os.path.splitext(audio_file)[0]
output_file = f"{base_name}.txt"

# Save to a text file
with open("transcription.txt", "w", encoding="utf-8") as f:

    f.write(result["text"])
