import ffmpeg
from time import perf_counter

def convert_wav_to_mp3(input_file, output_file):
    """
    Converts a WAV audio file to MP3 format using ffmpeg-python.

    Args:
        input_file (str): The path to the input WAV file.
        output_file (str): The desired path for the output MP3 file.
    """
    # ffmpeg-python handles the underlying ffmpeg process.
    # It will raise an ffmpeg.Error if ffmpeg is not found or if the command fails.

    start = perf_counter()
    try:
        (
            ffmpeg
            .input(input_file)
            .output(output_file, acodec='libmp3lame', audio_bitrate='192k')
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
        elapsed = perf_counter() - start
        print(f"Conversion successful: '{input_file}' converted to '{output_file}'")
        print(f"⏰ Elapsed time: {elapsed:.3f} seconds")
    except ffmpeg.Error as e:
        elapsed = perf_counter() - start
        print(f"Error during conversion (after {elapsed:.3f} seconds):")
        print(f"Stdout: {e.stdout.decode('utf8') if e.stdout else 'N/A'}")
        print(f"Stderr: {e.stderr.decode('utf8') if e.stderr else 'N/A'}")
    except FileNotFoundError:
        print("Error: The 'ffmpeg' executable was not found.")
        print("Please ensure FFmpeg is installed on your system and its location is in your system's PATH environmental variable.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Example usage:
    input_wav = "./media/Valicha notas.wav"
    output_mp3 = "./media/Valicha notas.mp3"

    convert_wav_to_mp3(input_wav, output_mp3)