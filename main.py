import tetris_ai.ga as ga
import tetris_ai.tetris_base as game
import tetris_ai.tetris_ai as ai
import tetris_ai.analyser as analyser
import matplotlib.pyplot as plt
import argparse
import copy
import time
import numpy as np


def main(no_show_game, num_gen=10, num_pop=10, num_exp=10,
         gap=0.1, mutation_rate=0.1, crossover_rate=0.8):
    # GENERAL CONFIG
    GAME_SPEED = 10000000
    NUM_GEN = num_gen
    NUM_POP = num_pop
    NUM_EXP = num_exp
    GAP = gap
    NUM_CHILD = round(NUM_POP*GAP)
    MUTATION_RATE = mutation_rate
    CROSSOVER_RATE = crossover_rate
    MAX_SCORE = 1000000

    genetic_alg = ga.GA(NUM_POP)

    best_chromos = []
    best_pop_score = []
    avg_pop_gen = []

    # Define datasets
    experiments = []
    best_chromo = []

    # Initialize population
    init_pop = ga.GA(NUM_POP)

    for e in range(NUM_EXP):
        # Make a copy from initial population so that we can run all experiments
        # with the same initial population
        pop = copy.deepcopy(init_pop)

        # Initialize generation list
        generations = []

        for g in range(NUM_GEN):
            print(' \n')
            print(f' - - - - Exp: {e}\t Generation: {g} - - - - ')
            print(' \n')

            # Save generation
            generations.append(copy.deepcopy(pop))

            # Select chromosomes using roulette method
            selected_pop = pop.selection(pop.chromosomes, NUM_CHILD,
                                         type="roulette")
            # Apply crossover and mutation
            new_chromo = pop.operator(selected_pop,
                                      crossover="uniform",
                                      mutation="random",
                                      crossover_rate=CROSSOVER_RATE,
                                      mutation_rate=MUTATION_RATE)

            for i in range(NUM_CHILD):
                # Run the game for each chromosome
                game_state = ai.run_game(pop.chromosomes[i], GAME_SPEED,
                                         MAX_SCORE, no_show_game)
                # Calculate the fitness
                new_chromo[i].calc_fitness(game_state)

            # Insert new children in pop
            pop.replace(new_chromo)
            fitness = [chrom.score for chrom in pop.chromosomes]
            print(fitness)

            # Print population
            print(pop)

        # Save experiments results
        experiments.append(generations)

        # Plot results
    an = analyser.Analyser(experiments)

    with open('plots/info.txt', 'a') as fd:
        fd.write(
            f"num_pop={num_pop}_num_gen={num_gen}_num_exp={num_exp}_gap={gap}_crossover_rate={crossover_rate}_mutation_rate={mutation_rate}:\n")

    an.plot(type="best", num_pop=num_pop, num_exp=num_exp, num_gen=num_gen,
            crossover_rate=crossover_rate, mutation_rate=mutation_rate, gap=gap)
    an.plot(type="pop", num_pop=num_pop, num_exp=num_exp, num_gen=num_gen,
            crossover_rate=crossover_rate, mutation_rate=mutation_rate, gap=gap)
    weights = an.plot(type="mdf", num_pop=num_pop, num_exp=num_exp, num_gen=num_gen,
                      crossover_rate=crossover_rate, mutation_rate=mutation_rate, gap=gap, show_std=False)
    with open('plots/info.txt', 'a') as fd:
        fd.write(f"weights: {weights}\n")
    # Return the best choromosome from all generation and experiments
    return an.weights


if __name__ == "__main__":
    # Define argparse options
    parser = argparse.ArgumentParser(description="Tetris AI")
    parser.add_argument('--train',
                        action='store_true',
                        help='Whether or not to train the AI')
    parser.add_argument('--game',
                        action='store_true',
                        help='Run the base game without AI')
    parser.add_argument('--no-show',
                        action='store_true',
                        help='Whether to show the game')
    parser.add_argument('--num-gen', type=int, default=10,
                        help='number of generations, default 10')
    parser.add_argument('--num-pop', type=int, default=10,
                        help='number of population, default 10')
    parser.add_argument('--num-exp', type=int, default=10,
                        help='number of experiments, default 10')
    parser.add_argument('--gap', type=float, default=0.2,
                        help='gap, default 0.2')
    parser.add_argument('--mutation-rate', type=float, default=0.15,
                        help='default 0.15')
    parser.add_argument('--crossover-rate', type=float, default=0.8,
                        help='default 0.8')

    args = parser.parse_args()

    if (args.train):
        # Train the AI and after play the game with the get chromosome
        start = time.time()
        best_chromos = main(args.no_show, args.num_gen, args.num_pop,
                            args.num_exp, args.gap, args.mutation_rate, args.crossover_rate)
        end = time.time()
        with open('plots/info.txt', 'a') as fd:
            fd.write(
                f"time: {np.round(end-start,2)}s\n -----------------------------------------------------------------\n")
        # FIXME:
        #chromo       = ga.Chromosome(best_chromos)
        #ai.run_game(chromo, speed=500, max_score=200000, no_show=False)

    elif (args.game):
        # Just run the base game
        game.MANUAL_GAME = True
        game.main()

    else:
        # Run tetris AI with optimal weights
        # FIXME: Define the optimal weights
        optimal_weights = [-0.97, 5.47, -13.74, -0.73,  7.99, -0.86, -0.72]
        chromo = ga.Chromosome(optimal_weights)
        ai.run_game(chromo, speed=600, max_score=200000, no_show=False)
