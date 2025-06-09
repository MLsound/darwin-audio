# main_orchestrator.py
import os
import re
import logging
from compressor import convert_wav_to_mp3
from peaq import run_peaq

# SETUP PARAMETERS
verbose = True # If True, prints additional information
debug = False # If True, creates an output folder and adds an index to the output file name
add_idx = False # If True, adds an index to the output file name

idx = 0
processed = None

def printt(message,n=80, char='='):
    sep = char*int((n-len(message))/2)
    message = f' {message} '
    print(sep,message,sep)

def build_output(file: str) -> str:
    global idx, add_idx
    base, ext = os.path.splitext(file)
    if ext.lower() != '.wav':
        raise ValueError('El formato del archivo de origen debe ser .WAV')
    if debug:
        add_idx = True
        # Create a subfolder named 'output' in the same directory as the input file
        folder_path = os.path.dirname(file)
        output_folder = os.path.join(folder_path, "output")
        os.makedirs(output_folder, exist_ok=True)
        base = os.path.join(output_folder, os.path.basename(base))
    if add_idx:
        idx += 1
        return f"{base}{idx}.mp3"
    else:
        return f"{base}.mp3"

def process_audio(file: str, params: dict | None) -> float | None:
    global processed, logger
    input_path = file
    output_path = build_output(file)
    if logger: logger.debug(f"Output: {output_path}")
    if verbose: print(f"Processing: '{input_path}' -> '{output_path}'")
    if logger and verbose: (logging.getLogger(f"{logger.name}.orch")).debug('ORCHESTRATOR module loaded.')

    success, compress_time = convert_wav_to_mp3(input_path, output_path, params, verbose, log_file=logger) # (+) setup hyperparams

    if success:
        if verbose: print(f"⚙️  Orchestrator: Successfully converted '{input_path}' (Duration: {compress_time:.3f})")
        processed = output_path
        return compress_time
    else:
        print(f"⚙️  Orchestrator: Failed to convert '{input_path}'.")
        return None

def extract_values(metrics: list[str]) -> tuple[float|None,float|None]:

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


def get_file_size(file_path: str) -> int:
    """
    Measures the size of a file in bytes.

    Args:
        file_path (str): The full path to the file.

    Returns:
        int: The size of the file in bytes, or -1 if the file does not exist or an error occurs.
    """
    try:
        if not os.path.exists(file_path):
            print(f"Error: File not found at '{file_path}'")
            return -1
        
        size_in_bytes = os.path.getsize(file_path)
        return size_in_bytes
    except OSError as e:
        print(f"Error accessing file '{file_path}': {e}")
        return -1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evaluate(file, params, verbose_sdk: bool = True, debug_mode = False, log_file=None):
    global processed, verbose, debug, logger

    logger = log_file

    if debug_mode:
        debug = True
        print("⚠️  Debug mode enabled! All files will be saved in an 'output' folder with an index number.")

    verbose = verbose_sdk
    process_time = 0

    printt("Starting audio evaluation process")
    # (+) Recieve parameters to test
    if verbose: print("TESTING HYPERPARAMETERS:", params)

    print("\n🗜️  STEP 1 - AUDIO COMPRESSION")
    compress_time = process_audio(file, params) # (+) config hyperparams
    if compress_time: process_time += compress_time

    print("\n🌡️  STEP 2 - QUALITY EVALUATION")
    result, metrics_time = run_peaq(file, processed, verbose, log_file=logger)
    if metrics_time: process_time += metrics_time

    # Adapt output format for extracting metrics
    if result: result_values = result.split("\n")
    result_values = [value for value in result_values if value.strip()]
    objective_difference_grade, distortion_index = extract_values(result_values)

    # Measure file size
    measured_size = get_file_size(processed)
    if measured_size == -1:
        print(f"It wasn't possible to measure file size for {processed}")
        size_output_file = None
    else:
        size_output_file = measured_size

    metrics = {
        'time': process_time,
        'size': size_output_file,
        'peaq': objective_difference_grade,
        'im': distortion_index,
    }
    if logger: logger.info(f" >> Metrics: {metrics}")

    print("\nMETRICS:")
    print(f" - File size: {metrics['size']/ 1024:.2f} KB")
    print(f" - Objective Difference Grade: {metrics['peaq']}")
    print(f" - Distortion Index: {metrics['im']}")


    print(f"\n⏰ Process time: {process_time} seconds")
    printt("Finished evaluation")
    print()
    return metrics

if __name__ == "__main__":
    input_wav = "./media/Valicha notas.wav"
    #input_wav = "./media/test.wav"

    # params = {
    #     'ar': '48000',
    #     'sample_fmt': 'fltp',
    #     'aq': '1',
    #     'audio_bitrate': '192'
    # }
    params = [
        {'ar': '22050'},
        {'b:a':'320k'},
        {'ar': '22050','b:a':'320k'},
    ]
    for param in params:
        metrics = evaluate(input_wav, param)
        print(metrics)