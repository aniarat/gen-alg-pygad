import logging

import pygad
import numpy
import benchmark_functions as bf
import matplotlib.pyplot as plt
import os

from Consts.enums import FunctionsOptions, MinMax, CrossingMethodsDec
from Helpers.crossingMethodsBin import TestCrossover
from Helpers.crossingMethodsDec import SingleArithmeticalCrossover, ArithmeticalCrossover, LinearCrossover, \
    BlendCrossoverAlfaBeta, BlendCrossoverAlfa, AverageCrossover, SimpleCrossover, RandomCrossover
from Helpers.mutationMethods import GaussMutation
from Helpers.plotsNFiles import make_plot, save_to_file

############ DZIESIETNA ############

num_genes = 2  #Liczba wymiarów

## FLAGI
func_enum = FunctionsOptions.RASTRIGIN  #Tutaj wybieramy funkcje do optymalizacji
func_min_max = MinMax.MIN  #Tutaj wybieramy czy liczymy maximum czy minimim
selected_crossover = CrossingMethodsDec.RANDOM  #Tutaj wybieramy funkcje crossover
parent_selection_type = "tournament"  #(rws)(random)
#Przypisanie parametrów
num_generations = 80
sol_per_pop = 80
num_parents_mating = 50

func = bf.Rastrigin(n_dimensions=num_genes) \
    if func_enum == FunctionsOptions.RASTRIGIN \
    else bf.Schwefel(n_dimensions=num_genes)

decode_start = func.suggested_bounds()[0][0]  #zakres początkowy w szukanej funkcji
decode_end = func.suggested_bounds()[1][0]  ##zakres końcowy w szukanej funkcji
# mutation_type = "swap"  #(random)(None)(swap)(inversion)(adaptive)
mutation_type = GaussMutation(num_genes, 0, 1, decode_start, decode_end, 0.2).mutate
mutation_type_name = "GaussMutation" #potrzebne do wykresu


# crossover_type = "uniform"  #(single_point)(two_points)(uniform)
match (selected_crossover):  #przypisanie własnych funckji crossover
    case CrossingMethodsDec.TEST:
        crossover_type = TestCrossover
    case CrossingMethodsDec.SINGLE_POINT_ARITHMETIC:
        crossover_type = SingleArithmeticalCrossover
    case CrossingMethodsDec.ARITHMETIC:
        crossover_type = ArithmeticalCrossover
    case CrossingMethodsDec.LINEAR:
        crossover_type = LinearCrossover(func).crossover
    case CrossingMethodsDec.BLEND_ALFA_BETA:
        crossover_type = BlendCrossoverAlfaBeta
    case CrossingMethodsDec.BLEND_ALFA:
        crossover_type = BlendCrossoverAlfa
    case CrossingMethodsDec.AVERAGE:
        crossover_type = AverageCrossover
    case CrossingMethodsDec.SIMPLE:
        crossover_type = SimpleCrossover
    case CrossingMethodsDec.RANDOM:
        crossover_type = RandomCrossover


def fitness_func_min(ga_instance, solution,
                     solution_idx):
    return 1. / func(solution)


def fitness_func_max(ga_instance, solution,
                     solution_idx):
    return func(solution)


fitness_function = fitness_func_min if func_min_max == MinMax.MIN else fitness_func_max

##(start) Parametry gwarantujące nam rozwiązanie binarne
init_range_low = decode_start
init_range_high = decode_end
mutation_num_genes = 2
gene_type = "int"

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
    ga_instance.logger.info("Individual    = {solution}".format(
        solution=repr(solution)))

    tmp = [1. / x for x in ga_instance.last_generation_fitness]  #ponownie odwrotność by zrobić sobie dobre statystyki

    ga_instance.logger.info("Min    = {min}".format(min=numpy.min(tmp)))
    ga_instance.logger.info("Max    = {max}".format(max=numpy.max(tmp)))
    ga_instance.logger.info("Average    = {average}".format(average=numpy.average(tmp)))
    ga_instance.logger.info("Std    = {std}".format(std=numpy.std(tmp)))
    ga_instance.logger.info("\r\n")

    avg_fitness.append(numpy.average(tmp))
    std_fitness.append(numpy.std(tmp))

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
                       gene_type=float,
                       random_mutation_max_val=1,
                       random_mutation_min_val=0,
                       logger=logger,
                       on_generation=on_generation,
                       parallel_processing=['thread', 4])
ga_instance.run()

best = ga_instance.best_solution()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best specimen : {solution}".format(
    solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=1. / solution_fitness))

# sztuczka: odwracamy my narysował nam się oczekiwany wykres dla problemu minimalizacji
ga_instance.best_solutions_fitness = [1. / x for x in ga_instance.best_solutions_fitness]
ga_instance.plot_fitness()

statistics = [ga_instance.best_solutions_fitness, avg_fitness, std_fitness]
labels = ['Best Fitness','Average Fitness', 'Standard Deviation']
# colors = ['lightcoral', 'lightseagreen', 'indigo']

for stat, label in zip(statistics, labels):
    make_plot(
        values = stat,
        file_name = os.path.join('Plots', f'Dec{label.replace(" ", "")}'),
        title = label,
        mutation_type = mutation_type_name,
        crossover_type = selected_crossover.name,
        selection_type = parent_selection_type
    )


