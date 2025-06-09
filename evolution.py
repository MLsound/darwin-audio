import os
import random as rnd
import numpy as np
from deap import base, creator, tools, algorithms
from orchestrator import evaluate, printt
import logging
from logger import setup_logger
import sys

# INITIAL SETUP:
# Input WAV file path (any music file in WAV format)
#input_wav = "./media/Valicha notas.wav"
input_wav = "./media/test.wav"

algo = "NSGA-II" 
verbose = False # Set to True for detailed output
debug = False # Set to True for debugging mode, which saves outputs in an 'output' folder
count = 1 # Counter for evaluations, used for tracking

# Setup the logger for the evolutionary algorithm
logger = setup_logger(algo, log_file=f"logs/evolution_{algo}.log",
                    level=logging.DEBUG if debug else logging.INFO, console_output=verbose)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# FOR TESTING PURPOSES:
# Uncomment the following lines to simulate the evaluation function and comment import from orchestrator.py

# def evaluate(file, params):
#     """
#     Simulated evaluation function that mimics the behavior of evaluating audio files.
#     This is a placeholder for the actual evaluation logic.
#     Args:
#         file (str): Path to the audio file to evaluate.
#         params (dict): Parameters for the evaluation.
#     Returns:
#         dict: Simulated metrics including size, PEAQ score, distortion index, and processing time.
#     """
#     print(f"Simulating evaluation for file: {file} with params: {params}")
#     # Simulate some metrics for demonstration purposes
#     return {
#         'size': rnd.randint(1000000, 5000000),  # Simulated file size in bytes
#         'peaq': rnd.uniform(-4.0, 0.0),         # Simulated PEAQ score in range -4 to 0
#         'im': rnd.uniform(-4.0, 0.0),           # Simulated Distortion Index in range -4 to 0
#         'time': rnd.uniform(0.1, 2.0)            # Simulated processing time in seconds
#     }

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

# --- Evolutionary Algorithm Parameters ---
POPULATION_SIZE = 50
MAX_GENERATIONS = 20
P_CROSSOVER = 0.8  # Probabilidad de cruce
P_MUTATION = 0.1   # Probabilidad de mutación

# --- Gene Definitions (Hyperparameters) ---

# 1. Audio Sample Rate (ar)
# Common values for LAME, can be restricted if needed.
# For simplicity, we'll evolve it as an integer and convert to string.
#LOW_SR, UP_SR = 11025, 48000 # Example range, you might want specific discrete choices
# Or, if you want only discrete choices:
SR_CHOICES = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000]
LOW_SR_IDX, UP_SR_IDX = 0, len(SR_CHOICES) - 1 # Evolves an index

# 2. Sample Format (sample_fmt) - Categorical
# Mapping integers to sample format strings
SAMPLE_FMT_MAP = {
    0: 's16p',
    1: 's32p',
    2: 'fltp'
}

LOW_SF, UP_SF = 0, len(SAMPLE_FMT_MAP) - 1 # Integer representation for evolution
LOW_BITRATE, UP_BITRATE = 32, 320 # in kbps (e.g., for audio_bitrate and abr)

# 3. LAME Compression Level (compression_level)
LOW_CL, UP_CL = 0, 9 # 0 (best quality/least compression) to 9 (highest compression)

# 4. Reservoir (reservoir) - Boolean (0 or 1)
LOW_RES, UP_RES = 0, 1 # 0 for off, 1 for on

# 5. Encoding Mode Selector (audio_bitrate, aq, or abr)
# 0: audio_bitrate (CBR)
# 1: aq (VBR Quality)
# 2: abr (Average Bitrate)
LOW_EM, UP_EM = 0, 2 # Encoding mode index
# Specific ranges for audio_bitrate (CBR/ABR) and aq (VBR quality) for clamping
LOW_QUAL, UP_QUAL = 0.0, 9.0     # for 'aq' (VBR quality, 0 is best, 9 is worst)

# 6. Value for the selected encoding mode
# This gene needs a wide range to cover bitrates (e.g., 32k-320k) and quality values (0-9).
# We'll treat it as a float and then convert/round as needed based on the encoding mode.
LOW_MODE_VAL, UP_MODE_VAL = 0.0, 320.0 # Max bitrate for CBR, max quality value for aq/abr (adjusted)


# Total Number of Dimensions/Genes
N_DIM = 6 # (ar, sample_fmt, compression_level, reservoir, encoding_mode, mode_value)

# --- DEAP Configuration ---
# Define the objectives: minimize size, maximize PEAQ, maximize Distortion Index, MINIMIZE PROCESSING TIME
# weights=(-1.0, 1.0, 1.0, -1.0) means:
# - minimize the first value (file_size)
# - maximize the second value (peaq_score)
# - maximize the third value (distortion_index)
# - minimize the fourth value (processing_time)
# creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, 1.0, -1.0))
creator.create("FitnessMulti", base.Fitness, weights=(-0.5, 1, 0.3, -0.05))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Gene Generators
toolbox.register("attr_samplerate_idx", rnd.randint, LOW_SR_IDX, UP_SR_IDX) # Generates an index
toolbox.register("attr_samplefmt", rnd.randint, LOW_SF, UP_SF) # Evolves an integer
toolbox.register("attr_compression_level", rnd.randint, LOW_CL, UP_CL)
toolbox.register("attr_reservoir", rnd.randint, LOW_RES, UP_RES)
toolbox.register("attr_encoding_mode", rnd.randint, LOW_EM, UP_EM) # Evolves 0, 1, or 2
toolbox.register("attr_mode_value", rnd.uniform, LOW_MODE_VAL, UP_MODE_VAL) # Wide range for mode value

# Individual and Population Generator
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_samplerate_idx,
                  toolbox.attr_samplefmt,
                  toolbox.attr_compression_level,
                  toolbox.attr_reservoir,
                  toolbox.attr_encoding_mode,
                  toolbox.attr_mode_value), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Genetic Operators
toolbox.register("select", tools.selNSGA2)

# Bounds for crossover and mutation (must match N_DIM)
# Ensure they are in the order of gene generation
# For categorical or discrete genes, the bounded operators will still treat them as continuous
# and then we'll round/map them in the evaluate function.
LOW_BOUNDS = [LOW_SR_IDX, LOW_SF, LOW_CL, LOW_RES, LOW_EM, LOW_MODE_VAL]
UP_BOUNDS = [UP_SR_IDX, UP_SF, UP_CL, UP_RES, UP_EM, UP_MODE_VAL]

toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                 low=LOW_BOUNDS, up=UP_BOUNDS, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded,
                  low=LOW_BOUNDS, up=UP_BOUNDS, eta=20.0, indpb=0.1)

# --- Evaluation Function ---
def evaluate_ffmpeg_params(individual, input_file_path):
    """
    Evaluates an individual by generating FFmpeg parameters,
    running FFmpeg, and computing fitness metrics.

    Args:
        individual (list): A DEAP individual representing the genes.
                           individual[0]: sample_rate (int)
                           individual[1]: sample_format (int, maps to string)
                           individual[2]: compression_level (int)
                           individual[3]: reservoir (int, 0 or 1)
                           individual[4]: encoding_mode (int, 0=CBR, 1=VBR-aq, 2=ABR)
                           individual[5]: mode_value (float/int for selected mode)
        input_file_path (str): Path to the input audio file.

    Returns:
        tuple: A tuple of fitness values (file_size, peaq_score, distortion_index).
    """
    global count
    print("\n") # Vertical space for clarity in output

    # Extract and process genes
    sample_rate_idx = int(round(individual[0]))
    # Clamp the index to ensure it's within valid range for SUPPORTED_SAMPLE_RATES
    sample_rate_idx = max(LOW_SR_IDX, min(UP_SR_IDX, sample_rate_idx))
    sample_rate = SR_CHOICES[sample_rate_idx] # Look up the actual sample rate
    # Map integer gene to actual sample format string
    sample_format_int = int(round(individual[1]))
    sample_format = SAMPLE_FMT_MAP.get(sample_format_int, 's32p') # Default to 's32p' if invalid

    compression_level = int(round(individual[2]))
    reservoir = int(round(individual[3])) # Will be 0 or 1

    encoding_mode = int(round(individual[4]))
    mode_value = individual[5] # Keep as float, will be cast/rounded based on mode

    ffmpeg_params = {
        #'acodec': 'libmp3lame',
        'ar': str(sample_rate),
        'sample_fmt': sample_format,
        'compression_level': str(compression_level),
        'reservoir': str(reservoir)
    }
    if verbose: print(f"\nFFmpeg Params (before mode handling): {ffmpeg_params}")

    # Handle mutually exclusive encoding modes
    if encoding_mode == 0: # Use audio_bitrate (CBR)
        bitrate_kbps = max(LOW_BITRATE, min(UP_BITRATE, int(round(mode_value)))) # Clamp to valid range
        ffmpeg_params['b:a'] = f"{bitrate_kbps}k"
    elif encoding_mode == 1: # Use aq (VBR Quality)
        quality_param = max(LOW_QUAL, min(UP_QUAL, round(mode_value))) # Clamp to valid range
        ffmpeg_params['aq'] = str(int(quality_param)) # aq expects integer
    elif encoding_mode == 2: # Use abr (Average Bitrate)
        bitrate_kbps = max(LOW_BITRATE, min(UP_BITRATE, int(round(mode_value)))) # Clamp to valid range
        ffmpeg_params['b:a'] = f"{bitrate_kbps}k"
        #abr_kbps = max(LOW_BITRATE, min(UP_BITRATE, int(round(mode_value)))) # Clamp to valid range
        ffmpeg_params['abr'] = str(True)
    # Else, no bitrate/quality parameter will be added, which might default to FFmpeg's own.
    # It's better to ensure one of these is always set for LAME.

    if verbose: print(f"\nFFmpeg Params (after mode handling): {ffmpeg_params}")

    file_size = float('inf') # Initialize with a large value for minimization
    peaq_score = 0.0         # Initialize with a low value for maximization
    distortion_index = 0.0   # Initialize with a low value for maximization

    # Print current individual and its ffmpeg parameters for debugging/tracking
    print(f"🧬 Evaluating Individual {count}º: {individual}")
    print(f"   FFmpeg Params: {ffmpeg_params}\n")
    logger.info(f"{count}º Individual: {ffmpeg_params}")
    logger.debug(f"Genetic parameters: {individual}")

    try:
        # SHOULD CALL orchestrator.py here
        metrics = evaluate(input_file_path, ffmpeg_params, verbose, debug_mode=debug, log_file=logger)
        count += 1 # Increment count for each evaluation

        if metrics:
            file_size = metrics['size'] / 1024.0
            peaq_score = metrics['peaq']
            distortion_index = metrics['im']
            processing_time = metrics['time']
    
    # Add error handling
    except Exception as e:
        logger.exception(f"General Error during evaluation for individual {individual}: {e}")
        print(f"General Error during evaluation for individual {individual}: {e}")
        file_size = float('inf')
        peaq_score = 0.0
        distortion_index = 0.0
        processing_time = float('inf') # Penalize heavily for errors
    
    return file_size, peaq_score, distortion_index, processing_time
    
# Register the evaluation function with the toolbox
toolbox.register("evaluate", evaluate_ffmpeg_params, input_file_path=input_wav)

# --- Main Evolutionary Algorithm Loop ---
def main_evolutionary_algorithm():
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.ParetoFront()  # Hall of fame for non-dominated solutions

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Run the evolutionary algorithm (NSGA-II))
    print("\n")
    printt(f"🪺  Starting Evolutionary Algorithm ({algo})  🦕", n=60, char="· ~ ")
    logger.info("GENETIC ALGORITHM STARTED")
    logger.info(f" Input: {input_wav}")
    logger.debug(f" Algorithm: {algo}")
    logger.info(f" Population Size: {POPULATION_SIZE}, Max Generations: {MAX_GENERATIONS}")
    logger.info(f" Crossover Probability: {P_CROSSOVER}, Mutation Probability: {P_MUTATION}")
    # Using eaMuPlusLambda as before, which is suitable for NSGA-II's non-dominated sorting.
    algorithms.eaMuPlusLambda(pop, toolbox, mu=POPULATION_SIZE, lambda_=POPULATION_SIZE,
                              cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)
    
    print("\n\n")
    printt("🏆 Best Non-Dominated Individuals (Pareto Front)")
    for ind in hof:
        # Reconstruct and print human-readable parameters for the best individuals
        sample_rate_idx = int(round(ind[0]))
        sample_rate_idx = max(LOW_SR_IDX, min(UP_SR_IDX, sample_rate_idx)) # Clamp for display
        sample_rate = SR_CHOICES[sample_rate_idx] # Look up actual value
        sample_format = SAMPLE_FMT_MAP.get(int(round(ind[1])), 's32p')
        compression_level = int(round(ind[2]))
        reservoir = int(round(ind[3]))
        encoding_mode_idx = int(round(ind[4]))
        mode_value_raw = ind[5]

        mode_str = ""
        if encoding_mode_idx == 0:
            mode_value_actual = max(LOW_BITRATE, min(UP_BITRATE, int(round(mode_value_raw))))
            mode_str = f"Bitrate: {mode_value_actual}k (CBR)"
        elif encoding_mode_idx == 1:
            mode_value_actual = max(LOW_QUAL, min(UP_QUAL, round(mode_value_raw)))
            mode_str = f"Quality: {int(mode_value_actual)} (VBR-aq)"
        elif encoding_mode_idx == 2:
            mode_value_actual = max(LOW_BITRATE, min(UP_BITRATE, int(round(mode_value_raw))))
            mode_str = f"Bitrate: {mode_value_actual}k (ABR True)"
        else:
            mode_str = "Invalid Mode"

        print(f"\nIndividual:")
        print(f"  Sample Rate: {sample_rate} Hz")
        print(f"  Sample Format: {sample_format}")
        print(f"  Compression Level: {compression_level}")
        print(f"  Reservoir: {bool(reservoir)}")
        print(f"  Encoding Mode: {mode_str}")
        print(f"  Raw Gene Values: {np.round(ind, 2)}")
        print(f"  Fitness (Size MB, PEAQ, Distortion, Time): {np.round(ind.fitness.values, 4)}")


    return pop, stats, hof

if __name__ == "__main__":

    # Check for debug flag
    if "-v" in sys.argv or "--verbose" in sys.argv:
        verbose = True
        if verbose: print("NOTE: Verbose mode enabled by flag.")

    if "-d" in sys.argv or "--debug" in sys.argv:
        debug = True
        logger.setLevel(logging.DEBUG)
        if verbose: print("NOTE: Debug mode enabled by flag.")

    #global input_wav
    input_wav = "./media/Valicha notas.wav"
    if not os.path.exists(input_wav):
        print(f"Error: Input file not found at {input_wav}")
        print("Please update 'input_wav' to a valid path.")
    else:
        try:
            init_time = np.datetime64('now')
            final_pop, final_stats, final_hof = main_evolutionary_algorithm()
            final_time = np.datetime64('now')
            
        except KeyboardInterrupt:
            print("\n\nExecution interrupted by user (KeyboardInterrupt).")
            logger.warning("Execution interrupted by user (KeyboardInterrupt).")
        except FileNotFoundError as fnf_error:
            print(f"\n\nFile not found error: {fnf_error}")
            logger.error(f"FileNotFoundError: {fnf_error}")
        except Exception as e:
            print(f"\n\nAn unexpected error occurred: {e}")
            logger.exception("Unexpected exception occurred")
        finally:
            # Shows total execution time
            total_time = final_time - init_time
            print(f"\n\n ⏱️ Total Execution Time: {total_time}")
            logger.info("EXECUTION FINISHED SUCCESSFULLY")
            logger.info(f" > Total Execution Time: {total_time}")