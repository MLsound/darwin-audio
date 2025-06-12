import ffmpeg
from time import perf_counter
import os
from logger import getLogger, get_handler

verbose = True # If True, prints additional information

def convert_wav_to_mp3(input_file: str,
                       output_file: str,
                       params: dict | None = None,
                       verbose_sdk: bool = True,
                       log_file = None) -> tuple[bool, float | None]:
    """
    Converts a WAV audio file to MP3 format using ffmpeg-python.

    Args:
        input_file (str): The path to the input WAV file.
        output_file (str): The desired path for the output MP3 file.

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if conversion was successful, False otherwise.
            - str: A message describing the outcome or error.
    """
    global verbose
    verbose = verbose_sdk

    if log_file:
        comp_logger = getLogger(log_file.name,'comp')
        comp_logger.addHandler(get_handler()) # Allowing console printing
        if verbose: comp_logger.debug('COMPRESSOR module loaded.')

    # ffmpeg-python handles the underlying ffmpeg process.
    # It will raise an ffmpeg.Error if ffmpeg is not found or if the command fails.

    start = perf_counter()

    if params is not None:
        if verbose: print(f"Processing with parameters: {params}")
    else:
        print("No additional parameters provided, using default settings.")
        params = {
            'b:a': '128k',
            'ar': '44100'
        }  # Default settings if no params are provided
        print(f"Processing with parameters: {params}")

    # Basic input validation
    if not os.path.exists(input_file):
        message = f"Error: Input file not found at '{input_file}'"
        print(message)
        return False, message
    
    try:
        (
            ffmpeg
            .input(input_file)
            .output(output_file, acodec='libmp3lame', **params)  # Unpack the dictionary here
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )

            # Apparently, if two conflicting parameters are specified,
            # ffmpeg-python will either use the latest one or raise an error.

            # Audio Arguments:
            # FIXED
            # acodec='libmp3lame': Specifies the MP3 codec for audio encoding. (fixed for MP3)
            # ac='1': Sets the number of audio channels to 1 (mono). (Fixed for this example)

            # TRAINABLE
            # ar='48000': Sets the audio sample rate to 48 kHz.
            # sample_fmt='s16': Sets the audio sample format to signed 16-bit integer.
            # audio_bitrate='192k': Sets the audio bitrate (CBR)
            # aq='1': Set the audio quality (VBR). This is an alias for -q:a.
            # abr='1': Set the audio quality (ABR).
            # compression_level='5': Sets the LAME compression level (0-9, where 9 is max compr).
            # reservoir='1': Allows LAME to save bits during simpler passages (1 by default).
        

            # Special option names:
            # Arguments with special names such as -qscale:v (variable bitrate),
            # -b:v (constant bitrate), etc. can be specified as a keyword-args dictionary as follows:
            # (
            #     ffmpeg
            #     .input('in.mp4')
            #     .output('out.mp4', **{'qscale:v': 3})
            #     .run()

            # String expressions:
            # Expressions to be interpreted by ffmpeg can be included as string parameters 
            # and reference any special ffmpeg variable names:
            # (
            #     ffmpeg
            #     .input('in.mp4')
            #     .filter('crop', 'in_w-2*10', 'in_h-2*20')
            #     .input('out.mp4')
            # )


            # General Arguments:
            # overwrite_output=True: Overwrites the output file if it already exists.
            # capture_stdout=True: Captures ffmpeg's standard output.
            # capture_stderr=True: Captures ffmpeg's standard error.
    
        elapsed: float = perf_counter() - start
        if verbose: comp_logger.debug(f"Conversion successful: '{input_file}' converted to '{output_file}'")
        if verbose: print(f"⏰ Elapsed time: {elapsed:.3f} seconds")
        return True, elapsed
    
    except ffmpeg.Error as e:
        elapsed = perf_counter() - start
        comp_logger.error(f"Error during conversion (after {elapsed:.3f} seconds):")
        comp_logger.debug(f"Stdout: {e.stdout.decode('utf8') if e.stdout else 'N/A'}")
        comp_logger.debug(f"Stderr: {e.stderr.decode('utf8') if e.stderr else 'N/A'}")
        return False, elapsed

    except FileNotFoundError:
        comp_logger.error("Error: The 'ffmpeg' executable was not found.")
        print("Please ensure FFmpeg is installed on your system and its location is in your system's PATH environmental variable.")
        return False, None

    except Exception as e:
        comp_logger.exception(f"An unexpected error occurred: {e}")
        return False, None

if __name__ == "__main__":
    # Example usage:
    #input_wav = "./media/Valicha.wav"
    #input_wav = "./media/Valicha.mp3"
    input_wav = "./media/test.wav"
    output_mp3 = "./media/test.mp3"
    
    convert_wav_to_mp3(input_wav, output_mp3)