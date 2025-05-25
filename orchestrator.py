# main_orchestrator.py
import os
from compressor import convert_wav_to_mp3 # Assuming the file is named audio_converter.py
from peaq import run_peaq

#processed = None

def printt(message,n=80):
    print()
    sep = "="*int((n-len(message))/2)
    message = f' {message} '
    print(sep,message,sep)

def build_output(file: str) -> str:
    base, ext = os.path.splitext(file)
    if ext.lower() != '.wav':
        raise ValueError('El formato del archivo de origen debe ser .WAV')
    return f"{base}.mp3"

def process_audio(file: str) -> float | None:
    global processed
    input_path = file
    output_path = build_output(file)

    print(f"Processing: '{input_path}' -> '{output_path}'")

    # Ensure the input path is absolute or correct relative to this script
    # For simplicity, let's assume paths are correct

    success, compress_time = convert_wav_to_mp3(input_path, output_path) # (+) setup hyperparams

    if success:
        print(f"⚙️  Orchestrator: Successfully converted '{input_path}' (Duration: {compress_time:.3f})")
        processed = output_path
        return compress_time
    else:
        print(f"⚙️  Orchestrator: Failed to convert '{input_path}'.")
        return None

def extract_values(metrics):
    import re
    objective_difference_grade = None
    distortion_index = None

    # Iterate through each string in the list
    for line in metrics:
        # Try to match Objective Difference Grade
        odg_match = re.search(r"Objective Difference Grade: (-?\d+\.\d+)", line)
        if odg_match:
            objective_difference_grade = float(odg_match.group(1))

        # Try to match Distortion Index
        di_match = re.search(r"Distortion Index: (-?\d+\.\d+)", line)
        if di_match:
            distortion_index = float(di_match.group(1))

    return objective_difference_grade, distortion_index

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evaluate(file, params):
    process_time = 0
    printt("Starting audio evaluation process")
    # (+) Recieve parameters to test
    print("TESTING HYPERPARAMETERS:", params)

    print("\n🗜️  STEP 1 - AUDIO COMPRESSION:")
    compress_time = process_audio(file) # (+) config hyperparams
    if compress_time: process_time += compress_time

    print("\n🌡️  STEP 2 - QUALITY EVALUATION:")
    result, metrics_time = run_peaq(file,processed)
    if metrics_time: process_time += metrics_time

    # Adapt output format for extracting metrics
    if result: result_values = result.split("\n")
    result_values = [value for value in result_values if value.strip()]
    objective_difference_grade, distortion_index = extract_values(result_values)

    metrics = {
        'time': process_time,
        'size': None,
        'peaq': objective_difference_grade,
        'im': distortion_index,
    }

    print("METRICS:")
    print(f"File size: {metrics['size']}")
    print(f"Objective Difference Grade: {metrics['peaq']}")
    print(f"Distortion Index: {metrics['im']}")


    print(f"\n⏰ Process time: {process_time} seconds")
    printt("Finished evaluation")
    return metrics

if __name__ == "__main__":
    #input_wav = "./media/Valicha notas.wav"
    input_wav = "./media/test.wav"
    params = [None,None,None]
    evaluate(input_wav, params)