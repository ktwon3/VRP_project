import EasyGA
import random

# Create the Genetic algorithm
ga = EasyGA.GA()
ga.parent_ratio = 0.5
ga.chromosome_length = 4
ga.target_fitness_type = 'min'
M = [[1,1], [2,2], [3,1], [4,3]]

def is_it_5(chromosome):
    """A very simple case test function - If the chromosomes gene value is a 5 add one
     to the chromosomes overall fitness value."""
    # Overall fitness value
    fitness = 0
    # For each gene in the chromosome
    """for gene in chromosome.gene_list:
        # Check if its value = 5
        if (gene.value != 5):
            # If its value is 5 then add one to
            # the overal fitness of the chromosome.
            fitness += 1"""

    for gene_num in range(1, len(chromosome.gene_list)):
        a,b = chromosome.gene_list[gene_num-1].value, chromosome.gene_list[gene_num].value
        x1, y1, x2, y2 = M[a][0], M[a][1], M[b][0], M[b][1]
        fitness += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if len(chromosome.gene_list) != len(set(chromosome.gene_list)): fitness += 100000
    return fitness


ga.fitness_function_impl = is_it_5

# Create random genes from 0 to 10
ga.chromosome_impl = lambda: random.sample([0,1,2,3],4)

while ga.active():
    # Evolve only a certain number of generations
    ga.evolve(5)
    # Print the current generation
    ga.print_generation()
    # Print the best chromosome from that generations population
    ga.print_best_chromosome()
    print(ga.population.mating_pool)
    # If you want to show each population
    #ga.print_population()
    # To divide the print to make it easier to look at
    print('-'*75)

print(ga.population.chromosome_list)