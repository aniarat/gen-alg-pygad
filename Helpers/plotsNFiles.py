from matplotlib import pyplot as plt


def make_plot(values: list, file_name: str, title: str, y_scale: str = 'linear') -> None:
    plt.plot(values)
    plt.yscale(y_scale)
    plt.title(title)
    plt.xlabel('Epoka')
    plt.ylabel(file_name)
    plt.savefig(f'{file_name}.png')
    plt.figure()
    plt.close('all')


def save_to_file(values: list, spec: list, file_name: str) -> None:
    with open(f'./{file_name}.txt', 'w') as f:
        for i in range(len(values)):
            f.write(f'epoch {i+1}: f{spec[i]} = {values[i]}\n')
            i += 1
