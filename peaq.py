import os
import subprocess
from dotenv import load_dotenv

def run_peaq(ref_file, test_file, advanced=False):
    load_dotenv()

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

    command = ["peaq"]
    if advanced:
        command.append("--advanced")
    command.extend([ref_file, test_file])

    print(f"Running command: {' '.join(command)}")

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Stderr:")
            print(result.stderr)
    except FileNotFoundError:
        print(f"Error: 'peaq' command not found. Ensure it's in your PATH (either system-wide or via .env and this script).")
    except subprocess.CalledProcessError as e:
        print(f"Error running peaq command: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")

if __name__ == "__main__":
    ref_file_path = "/Users/alejandrolloveras/Documents/ESTUDIO/UBA/Materias/AE 1/Audio/media/Valicha notas.wav"
    test_file_path = "/Users/alejandrolloveras/Documents/ESTUDIO/UBA/Materias/AE 1/Audio/media/Valicha notas.mp3"

    run_peaq(ref_file_path, test_file_path)
    #run_peaq(ref_file_path, test_file_path, advanced=True)
