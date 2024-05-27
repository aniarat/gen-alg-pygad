from matplotlib import pyplot as plt

def make_plot(values: list, file_name: str, title: str, mutation_type: str, crossover_type: str, selection_type: str, num_genes: str, num_of_dimensions: str, num_generations: str, sol_per_pop: str, num_parents_mating: str,  y_scale: str = 'linear') -> None:
    plt.plot(values, label=title)
    plt.yscale(y_scale)
    plt.title(f'{title}\nMutation: {mutation_type}, Crossover: {crossover_type}, Selection: {selection_type}')
    plt.xlabel('Epoch')
    plt.ylabel(title)
    # plt.legend()
    plt.legend([f"num_genes={num_genes}", f"num_of_dimensions={num_of_dimensions}", f"num_generations={num_generations}", f"sol_per_pop={sol_per_pop}", f"num_parents_mating={num_parents_mating}"])
    plt.grid(True)
    plt.savefig(f'{file_name}.png')
    plt.show()
    plt.figure()
    plt.close('all')


def save_to_file(values: list, spec: list, file_name: str) -> None:
    with open(f'./{file_name}.txt', 'w') as f:
        for i in range(len(values)):
            f.write(f'epoch {i+1}: f{spec[i]} = {values[i]}\n')
            i += 1
