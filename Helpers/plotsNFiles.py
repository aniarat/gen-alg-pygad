from matplotlib import pyplot as plt

def make_plot(values: list, file_name: str, title: str, mutation_type: str, crossover_type: str, selection_type: str, y_scale: str = 'linear') -> None:
    plt.plot(values, label=title)
    plt.yscale(y_scale)
    plt.title(f'{title}\nMutation: {mutation_type}, Crossover: {crossover_type}, Selection: {selection_type}')
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.legend()
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
