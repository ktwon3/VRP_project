import utill
from VRP import My_GA
import random

ga = My_GA()
def chromosome_impl(ga):
    return [random.randint(1,20)]

ga.chromosome_impl = chromosome_impl
ga.initialize_population()
for chromosome in ga.population:
    chromosome.fitness = random.randint(0,10)
ga.print_population()

utill.costTofitenss(ga)
print(ga.real_fitness)




