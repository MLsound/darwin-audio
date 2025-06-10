import os
import subprocess
from dotenv import load_dotenv
from time import perf_counter
from logger import getLogger

# Load environment variables from .env file at the script level
# This ensures that if this module is imported, environment variables are loaded
# if the .env file is present in the context where the parent script is run
# or where this module resides.
load_dotenv()

verbose = True # If True, prints additional information

def _setup_environment():
    """
    Sets up necessary environment variables for GStreamer and PEAQ.
    This is called internally by run_peaq.
    """
    gstreamer_bin_path = os.getenv('GSTREAMER_BIN_PATH')
    gstreamer_pkgconfig_path = os.getenv('GSTREAMER_PKGCONFIG_PATH')
    gstreamer_aclocal_path = os.getenv('GSTREAMER_ACLOCAL_PATH')
    gstreamer_plugin_path = os.getenv('GSTREAMER_PLUGIN_PATH')

    if gstreamer_pkgconfig_path:
        os.environ['PKG_CONFIG_PATH'] = f"{gstreamer_pkgconfig_path}:{os.getenv('PKG_CONFIG_PATH', '')}"
    if gstreamer_aclocal_path:
        os.environ['ACLOCAL_PATH'] = f"{gstreamer_aclocal_path}:{os.getenv('ACLOCAL_PATH', '')}"
    if gstreamer_bin_path:
        os.environ['PATH'] = f"{gstreamer_bin_path}:{os.getenv('PATH', '')}"
    if gstreamer_plugin_path:
        os.environ['GST_PLUGIN_PATH'] = f"{gstreamer_plugin_path}:{os.getenv('GST_PLUGIN_PATH', '')}"

def run_peaq(ref_file: str, 
             test_file: str,
             verbose_sdk: bool = True,
             advanced: bool = False,
             log_file = None) -> tuple[subprocess.CompletedProcess | None, float | None]:
    """
    Runs the PEAQ command with the given reference and test audio files.

    Args:
        ref_file_path (str): Path to the reference audio file.
        test_file_path (str): Path to the test audio file.
        advanced (bool, optional): Whether to use the '--advanced' PEAQ option.
                                   Defaults to False.

    Returns:
        Optional[Tuple[subprocess.CompletedProcess, float]]:
            A tuple containing the subprocess result object and the elapsed time in seconds.
            Returns None if 'peaq' command is not found or if a CalledProcessError occurs.
    """
    _setup_environment() # Ensure environment variables are set for each call

    global verbose
    verbose = verbose_sdk

    if log_file:
        #peaq_logger = logging.getLogger(f"{log_file.name}.peaq")
        peaq_logger = getLogger(log_file.name,'peaq')
        if verbose: peaq_logger.debug('PEAQ module loaded.')

    command = ["peaq"]
    if advanced:
        command.append("--advanced")
    command.extend([ref_file, test_file])

    if verbose: print(f">> Running command: {' '.join(command)}")

    start_time = perf_counter()
    try:
        # It decodes stdout and stderr using the default locale encoding.
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        elapsed = perf_counter() - start_time

        if verbose: print(f"⏰ Elapsed time: {elapsed:.3f} seconds")
        metrics_value = result.stdout
        if result.stderr:
            print("Stderr:")
            print(result.stderr)

        return metrics_value, elapsed

    except FileNotFoundError:
        print(f"Error: 'peaq' command not found. Ensure it's in your PATH (either system-wide or via .env and this script).")
        return None, None
    except subprocess.CalledProcessError as e:
        elapsed = perf_counter() - start_time
        print(f"Error running PEAQ command: {e}")
        print(f"- Command: '{e.cmd}'")
        print(f"- Return code: {e.returncode}")
        print(f"- Stdout: {e.stdout}")
        print(f"- Stderr: {e.stderr}")
        print(f"Elapsed time: {elapsed:.3f} seconds")
        return None, elapsed

if __name__ == "__main__":
    ref_file_path = "./media/Valicha.wav"
    test_file_path = "./media/Valicha.mp3"

    run_peaq(ref_file_path, test_file_path)
    #run_peaq(ref_file_path, test_file_path, advanced=True)
