# 🧬🎵 Multi-Objective Optimization for Audio Compression
This project aims to optimize the compression of WAV audio files to MP3 using Perceptual Evaluation of Audio Quality (PEAQ) and an Improved Metric of Informational Masking (IM), both audioperception metrics based on a human ear model. The primary goal is to minimize file size while retaining the highest possible perceptual quality.

This is achieved through an automated framework that employs an evolutionary algorithm. This algorithm intelligently explores various FFmpeg (specifically, LAME MP3) encoding settings to identify optimal trade-offs. The framework orchestrates the entire process: it handles audio compression, evaluates the resulting audio quality using the specified metrics, and then leverages genetic algorithms to converge on a set of Pareto-optimal compression configurations specifically tailored to the input audio content. This approach helps users achieve a balance between file efficiency and listening experience.

## 📄 Documentation

A comprehensive report detailing the methodology, implementation, and results of this project is available in both Spanish and English. 
* **Spanish Report:** [Informe de Optimización Multiobjetivo en Audio](./Informe_Optimizacion_Audio_ES.pdf)
* **English Report:** [Multi-Objective Audio Optimization Report](./Report_Audio_Optimization_EN.pdf)

## 🎓 Project Context

This framework was developed as the **final project for the "Evolutionary Algorithms" course** in the **Master's in Artificial Intelligence program** at the Faculty of Engineering, University of Buenos Aires (FIUBA, UBA). It was completed under the guidance of **Prof. Miguel Augusto Azar** (2025).


## 📦 Requirements

❕Before running this project, you'll first need to successfully install the GST Peaq plugin for the GStreamer framework. That's no easy task, so I prepared a step-by-step guide to help with that purpose.

▸ [***Installation guide for GST Peaq & GStreamer***](./GStreamer_install.md)

---

The requirements could be installed through either `environment.yml` or `requirements.txt`

For system-level tools, you'll need to use your operating system's package manager to install them. You should use `brew install` for macOS or `sudo apt install` for Linux. Installing these tools on Windows can be more complex.

- Python 3.10+
- Main libraries:
  - `ffmpeg-python`
  - `pydub`
  - `deap`

- Specific libraries:
  - `subprocess`
  - `python-dotenv`
  - `re`

- Common libraries:
  - `random`
  - `numpy`
  - `time`
  - `os`

- System Dependencies & Build Tools:
  - `ffmpeg`
  - `pkg-config`
  - `autoconf`
  - `automake`
  - `libtool`
  - `gtk-doc`
  - `make`

## 📂 Project Structure

* `evolution.py`: This project now includes multiple scripts for different multi-objective evolutionary algorithms. `evolution.py` (and its numbered variants like `evolution2.py`, `evolution3.py`, `evolution4.py`, `evolution5.py`) implements the core multi-objective evolutionary algorithms (NSGA-II, NSGA-III, SPEA2, PAES, and MOEA-D, respectively) using `DEAP`. These scripts define the genes (audio compression parameters) and orchestrate the evaluation of individuals for their specific algorithms.
* `orchestrator.py`: Acts as the central hub for the evaluation process. It calls `compressor.py` to compress audio and `peaq.py` to assess the quality of the compressed file. It also measures file size and total processing time.
* `compressor.py`: Handles the actual audio compression from WAV to MP3 using `ffmpeg-python`, applying the parameters specified by the evolutionary algorithm.
* `peaq.py`: Provides an interface to the external `peaq` command-line tool, used for objective perceptual audio quality measurements. It also manages environment variables for correct tool execution.
* `logger.py`: A module dedicated to the centralized configuration and management of loggers. It defines the setup_logger function to create and configure loggers with outputs to files and/or the console, enabling unified and consistent logging of events, debugging, and important information throughout the entire project.
* `audiofile_analysis.ipynb`: A Jupyter Notebook for quick analysis of audio file properties (e.g., sample rate, bitrate) using `pydub`.
* `graph.ipynb`: A Jupyter Notebook for visualizing the results of the evolutionary algorithms, such as Pareto fronts and fitness landscapes.
* `./media/`: Directory intended for input WAV files and generated MP3 output files.
* `./logs/`: Contains log files (.log) detailing the execution flow, debugging information, and significant events of the evolutionary algorithm. Each run generates a timestamped log file.
* `./history/`: Stores CSV files (.csv) containing the full evolutionary history, including parameters, fitness values (file size, PEAQ score, distortion index), and processing time for each evaluated individual. Each run generates a timestamped CSV for distinct record-keeping.
* `./results/`: Holds processed output or final outcome files (e.g., CSVs of the best-performing individuals or specific analysis results) from the evolutionary runs (HOF).

## 📚 References

This project is built upon established research in audio quality assessment and evolutionary algorithms.

* **PEAQ (Perceptual Evaluation of Audio Quality)**:
    * The ITU-R Recommendation BS.1387 standard.
    * **Paper:** "An Examination and Interpretation of ITU-R BS.1387: Perceptual Evaluation of Audio Quality" by Peter Kabal (McGill University).

      ▸ [*Link to the article*](https://www.mmsp.ece.mcgill.ca/Documents/Reports/2002/KabalR2002v2.pdf)
* **IM (Improved Metric of Informational Masking)**:
    * Enhanced prediction of perceived audio quality (especially music)
    * **Paper:** "An Improved Metric of Informational Masking for Perceptual Audio Quality Measurement" by Pablo M. Delgado and Jürgen Herre.

      ▸ [*Link to the article*](https://arxiv.org/abs/2307.06656)

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## 📄 License

This project is open-sourced under the [MIT License](https://mit-license.org/).