import numpy as np
import matplotlib.pyplot as plt


class Analyser():
    def __init__(self, data):
        self.data = data
        self.best = 0
        self.weights = []

    def plot(self, type, num_pop, num_gen, num_exp, gap, crossover_rate,
             mutation_rate, show_mean=True, show_std=True,
             show_chromosomes=True, results=True, save=True):
        """
        Args:
            type             : Select plot type: best or pop
            show_mean        : Whether to show mean
            show_std         : Whether to show std
            show_chromosomes : Whether to show chromosomes
            save             : Whether to save
        """
        plt_config = {"best": {"title": "Best person",
                               "label": "Mean value of the best person",
                               "xlabel": "Generations",
                               "ylabel": "Fitness"},
                      "pop": {"title": "Mean population",
                              "label": "Mean value of the mean population",
                              "xlabel": "Generations",
                              "ylabel": "Fitness"},
                      "mdf": {"title": "Diversity in a phenotype",
                              "label": "MDF",
                              "xlabel": "Generations",
                              "ylabel": "MDF"}}

        label = plt_config[type]["label"]
        title = plt_config[type]["title"]
        xlabel = plt_config[type]["xlabel"]
        ylabel = plt_config[type]["ylabel"]

        plt.figure()

        experiment = []

        for i, exp in enumerate(self.data):
            generation = []
            N_gen = len(exp)

            for gen in exp:
                fitness = [chrom.score for chrom in gen.chromosomes]
                if (type == "best"):
                    value = np.amax(fitness)
                elif (type == "pop"):
                    value = np.mean(fitness)
                elif (type == "mdf"):
                    best = np.amax(fitness)
                    pop = np.mean(fitness)
                    value = pop/best
                else:
                    raise ValueError(f"Type {type} not defined")

                generation.append(value)

            experiment.append(generation)

            # FIXME
            best = np.amax(experiment)
            if (best > self.best):
                self.best = best
                i_best = np.argmax(experiment)
                try:
                    self.weights = gen.chromosomes[i_best].weights
                except IndexError:
                    print("FIXME: Index error analyser")

            if (show_chromosomes):
                plt.plot(np.arange(1, N_gen+1), generation, marker='o',
                         linewidth=0.5, markersize=1)

        if (show_mean):
            mean = []
            std = []

            for i in range(0, N_gen):
                exp = np.array(experiment)
                mean.append(np.mean(exp[:, i]))
                std.append(np.std(exp[:, i]))

            mean = np.array(mean)
            std = np.array(std)

            plt.plot(range(1, N_gen+1), mean, marker='o',
                     linewidth=1.5, markersize=2, color='black',
                     label=label)

            if (show_std):
                plt.fill_between(range(1, N_gen+1), mean-std, mean+std,
                                 linestyle='-', alpha=0.4, color='black',
                                 label='standard deviation')
            if (results):
                result = np.amax(mean)
                print(f"{label}: result = {result}")
                with open('plots/info.txt', 'a') as fd:
                    fd.write(f"{label} fitness = {result}\n")
                    # fd.write(f"weights: {self.weights}\n")

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ticklabel_format(useOffset=False)
        plt.legend()

        if (save):
            gap = str(gap).replace(".", ",")
            crossover_rate = str(crossover_rate).replace(".", ",")
            mutation_rate = str(mutation_rate).replace(".", ",")
            plt.savefig(
                f"plots/fitness_vs_gen_{type}_num_pop={num_pop}_num_gen={num_gen}_num_exp={num_exp}_gap={gap}_crossover_rate={crossover_rate}_mutation_rate={mutation_rate}", dpi=300)
            return self.weights
        else:
            plt.show()
