from deap import base, creator, tools, algorithms
import random as rnd
import numpy as np
from orchestrator import evaluate
#metrics = evaluate(file, params)

# --- Parámetros del Algoritmo Evolutivo ---
POPULATION_SIZE = 50
MAX_GENERATIONS = 20
P_CROSSOVER = 0.9  # Probabilidad de cruce
P_MUTATION = 0.2   # Probabilidad de mutación

# --- Definición de Genes (Hiperparámetros) ---
# Ejemplo: [bitrate_kbps, quality_param_float]
# Bitrate: entero entre 64 y 320 kbps
# Quality_param: flotante entre 1.0 y 10.0

# Límites para los genes
LOW_BITRATE, UP_BITRATE = 64, 320
LOW_QUAL, UP_QUAL = 1.0, 10.0
N_DIM = 2 # Número de dimensiones/genes

# --- Configuración de DEAP ---

# Definir los objetivos: minimizar tamaño, maximizar PEAQ, maximizar Distortion Index  !!
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Generador de atributos (genes)
toolbox.register("attr_bitrate", rnd.randint, LOW_BITRATE, UP_BITRATE)
toolbox.register("attr_quality", rnd.uniform, LOW_QUAL, UP_QUAL)

# Generador de individuos y población
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_bitrate, toolbox.attr_quality), n=1) # n=1 porque initCycle toma una tupla de generadores
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Registrar la función de evaluación
toolbox.register("evaluate", evaluate) # Nuestra función definida arriba

# Registrar los operadores genéticos (NSGA-II es bueno para multiobjetivo)
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=[LOW_BITRATE, LOW_QUAL], up=[UP_BITRATE, UP_QUAL], eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[LOW_BITRATE, LOW_QUAL], up=[UP_BITRATE, UP_QUAL], eta=20.0, indpb=0.1)


def main_evolutionary_algorithm():
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.ParetoFront() # Hall of fame para guardar las mejores soluciones no dominadas

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Ejecutar el algoritmo evolutivo (similar a eaMuPlusLambda o eaSimple, pero adaptado para NSGA-II)
    algorithms.eaMuPlusLambda(pop, toolbox, mu=POPULATION_SIZE, lambda_=POPULATION_SIZE,
                                cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    print("\n--- Mejores Individuos No Dominados (Frente de Pareto) ---")
    for ind in hof:
        print(f"Individuo: {ind}, Fitness: {ind.fitness.values}")
        # ind[0] -> bitrate, ind[1] -> quality_param
        # ind.fitness.values[0] -> file_size
        # ind.fitness.values[1] -> peaq_score
        # ind.fitness.values[2] -> distortion_index

    return pop, stats, hof

if __name__ == "__main__":
    input_wav = "./media/Valicha notas.wav"
    #input_wav = "./media/test.wav"
    main_evolutionary_algorithm(input_wav)