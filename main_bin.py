import logging

import pygad
import numpy
import matplotlib.pyplot as plt
import benchmark_functions as bf

from Consts.enums import FunctionsOptions, MinMax, CrossingMethodsBin
from Helpers.crossingMethodsBin import TestCrossover, SinglePointCrossover, TwoPointCrossover, ThreePointCrossover, \
    UniformCrossover, GrainCrossover, ScanningCrossover, PartialCopyCrossover, MultivariateCrossover
from Helpers.decimalBinaryMath import binary_to_decimal, split, get_actual_values
from Helpers.plotsNFiles import make_plot, save_to_file


############ BINARNA ############

num_genes = 48  #Długość osobnika * Liczba wymiarów
num_of_dimensions = 2  #Liczba wymiarów

## FLAGI
func_enum = FunctionsOptions.RASTRIGIN  #Tutaj wybieramy funkcje do optymalizacji
func_min_max = MinMax.MIN  #Tutaj wybieramy czy liczymy maximum czy minimim
selected_crossover = CrossingMethodsBin.MULTIVARIATE  #Tutaj wybieramy funkcje crossover
q = 3 #q dla MULTIVARIATE
parent_selection_type = "tournament"  #(rws)(random)
mutation_type = "swap"  #(random)(None)(swap)(inversion)(adaptive)

#Przypisanie parametrów
num_generations = 80
sol_per_pop = 80
num_parents_mating = 50


func = bf.Rastrigin(n_dimensions=num_of_dimensions) \
    if func_enum == FunctionsOptions.RASTRIGIN \
    else bf.Schwefel(n_dimensions=num_of_dimensions)

decode_start = func.suggested_bounds()[0][0]  #zakres początkowy w szukanej funkcji
decode_end = func.suggested_bounds()[1][0]  ##zakres końcowy w szukanej funkcji

# crossover_type = "uniform"  #(single_point)(two_points)(uniform)
match (selected_crossover):  #przypisanie własnych funckji crossover
    case CrossingMethodsBin.TEST:
        crossover_type = TestCrossover
    case CrossingMethodsBin.BUILD_IN_SINGLE_POINT:
        crossover_type = CrossingMethodsBin.BUILD_IN_SINGLE_POINT.value
    case CrossingMethodsBin.BUILD_IN_DOUBLE_POINT:
        crossover_type = CrossingMethodsBin.BUILD_IN_DOUBLE_POINT.value
    case CrossingMethodsBin.BUILD_IN_UNIFORM:
        crossover_type = CrossingMethodsBin.BUILD_IN_UNIFORM.value
    case CrossingMethodsBin.SINGLE_POINT:
        crossover_type = SinglePointCrossover
    case CrossingMethodsBin.DOUBLE_POINT:
        crossover_type = TwoPointCrossover
    case CrossingMethodsBin.UNIFORM:
        crossover_type = ThreePointCrossover
    case CrossingMethodsBin.UNIFORM:
        crossover_type = UniformCrossover
    case CrossingMethodsBin.GRAIN:
        crossover_type = GrainCrossover
    case CrossingMethodsBin.SCANNING:
        crossover_type = ScanningCrossover(num_of_dimensions).crossover
    case CrossingMethodsBin.PARTIAL:
        crossover_type = PartialCopyCrossover
    case CrossingMethodsBin.MULTIVARIATE:
        crossover_type = MultivariateCrossover(q).crossover

def fitness_func_min(ga_instance, solution,
                     solution_idx):
    actual_value = get_actual_values(solution, decode_start, decode_end, num_of_dimensions)
    return 1. / func(actual_value)


def fitness_func_max(ga_instance, solution,
                     solution_idx):
    actual_value = get_actual_values(solution, decode_start, decode_end, num_of_dimensions)
    return func(actual_value)


fitness_function = fitness_func_min if func_min_max == MinMax.MIN else fitness_func_max

##(start) Parametry gwarantujące nam rozwiązanie binarne
init_range_low = 0
init_range_high = 2
mutation_num_genes = 1
gene_type = "int"
##(end) Parametry gwarantujące nam rozwiązanie binarne

#Konfiguracja logowania

level = logging.DEBUG
name = 'logfile.txt'
logger = logging.getLogger(name)
logger.setLevel(level)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

avg_fitness = []
std_fitness = []

def on_generation(ga_instance):
    ga_instance.logger.info("Generation = {generation}".format(generation=ga_instance.generations_completed))
    solution, solution_fitness, solution_idx = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness)
    ga_instance.logger.info("Best    = {fitness}".format(fitness=solution_fitness))
    ga_instance.logger.info("Individual(2)    = {solution}".format(solution=list(split(solution, num_of_dimensions))))
    ga_instance.logger.info("Individual(10)    = {solution}".format(
        solution=repr(get_actual_values(solution, decode_start, decode_end, num_of_dimensions))))

    tmp = [1. / x for x in ga_instance.last_generation_fitness]  #ponownie odwrotność by zrobić sobie dobre statystyki

    avg_fitness.append(numpy.average(tmp))
    std_fitness.append(numpy.std(tmp))

    ga_instance.logger.info("Min    = {min}".format(min=numpy.min(tmp)))
    ga_instance.logger.info("Max    = {max}".format(max=numpy.max(tmp)))
    ga_instance.logger.info("Average    = {average}".format(average=numpy.average(tmp)))
    ga_instance.logger.info("Std    = {std}".format(std=numpy.std(tmp)))
    ga_instance.logger.info("\r\n")


#Właściwy algorytm genetyczny
ga_instance = pygad.GA(num_generations=num_generations,
                       sol_per_pop=sol_per_pop,
                       num_parents_mating=num_parents_mating,
                       num_genes=num_genes,
                       fitness_func=fitness_function,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       mutation_num_genes=mutation_num_genes,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       keep_elitism=1,
                       K_tournament=3,
                       gene_type=int,
                       random_mutation_max_val=1,
                       random_mutation_min_val=0,
                       logger=logger,
                       on_generation=on_generation,
                       parallel_processing=['thread', 4])
ga_instance.run()

best = ga_instance.best_solution()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best specimen(2) : {solution}".format(solution=list(split(solution, num_of_dimensions))))
print("Best specimen(10) : {solution}".format(
    solution=get_actual_values(solution, decode_start, decode_end, num_of_dimensions)))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=1. / solution_fitness))

# sztuczka: odwracamy my narysował nam się oczekiwany wykres dla problemu minimalizacji
ga_instance.best_solutions_fitness = [1. / x for x in ga_instance.best_solutions_fitness]
ga_instance.plot_fitness()

statistics = [ga_instance.best_solutions_fitness, avg_fitness, std_fitness]
labels = ['Best Fitness','Average Fitness', 'Standard Deviation']
# colors = ['lightcoral', 'lightseagreen', 'indigo']

for stat, label in zip(statistics, labels):
    make_plot(
        values=stat,
        file_name=f'Bin{label.replace(" ", "")}',
        title=label,
        mutation_type=mutation_type,
        crossover_type=selected_crossover.name,
        selection_type=parent_selection_type
    )



