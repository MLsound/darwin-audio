# main_orchestrator.py
import os
from compressor import convert_wav_to_mp3 # Assuming the file is named audio_converter.py

def printt(message,n=80):
    print()
    sep = "="*int((n-len(message))/2)
    message = f' {message} '
    print(sep,message,sep)

def build_output(file):
    base, ext = os.path.splitext(file)
    if ext.lower() != '.wav':
        raise ValueError('El formato del archivo de origen debe ser .WAV')
    return f"{base}.mp3"

def process_audio(file):

    input_path = file
    output_path = build_output(file)

    print(f"Processing: '{input_path}' -> '{output_path}'")

    # Ensure the input path is absolute or correct relative to this script
    # For simplicity, let's assume paths are correct

    success, process_time = convert_wav_to_mp3(input_path, output_path) # (+) setup hyperparams

    if success:
        print(f"⚙️  Orchestrator: Successfully converted '{input_path}' (Duration: {process_time:.3f})")
        # Add further post-processing steps here if needed
    else:
        print(f"⚙️  Orchestrator: Failed to convert '{input_path}'.")
        # Add error handling, logging, or retry logic here

def evaluate(file, params):
    printt("Starting audio evaluation process")
    # (+) Recieve parameters to test
    print("TESTING HYPERPARAMETERS:", params)

    print("\n🗜️  STEP 1 - AUDIO COMPRESSION:")
    process_audio(file) # (+) config hyperparams

    print("\n🌡️  STEP 2 - QUALITY EVALUATION:")

    metrics = {
        'time': None,
        'size': None,
        'peaq': None,
        'im': None,
    }

    print(metrics)

    printt("Finished evaluation")
    return metrics

if __name__ == "__main__":
    #input_wav = "./media/Valicha notas.wav"
    input_wav = "./media/test.wav"
    params = [None,None,None]
    evaluate(input_wav, params)