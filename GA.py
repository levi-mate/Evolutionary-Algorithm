import random
import copy
import matplotlib.pyplot as plt
import time
# Static parameters
N = 20

# For Dixon-Price function
#MIN = -10.0
#MAX = 10.0

# For Zakharov function
#MIN = -5.0
#MAX = 10.0

# For Rotated Hyper-Ellipsoid function
MIN = -65.536
MAX = 65.536

# Dynamic parameters for population size, number of generations, runs, elites, local search iterations, tournament size, crossover and mutation values
RUNS = 10
P = 50
GENS = 50
T = 5
CROSSOVER = 4
#CROSSOVER = 0.5
MUTRATE = 0.756
MUTSTEP = 0.690
ELITE = int(P * 0.05)
LOCALITER = 20

save_data = "GA_results.txt"

class individual:
    def __init__(self):
        self.gene = [0] * N
        self.fitness = 0

# Fitness functions, simply rename the one to use test_function, and name the other one something else
# Dixon-Price
def test_functionA(ind):
    utility=0
    x = ind.gene
    for i in range(1, N):
       utility += (i * (((2 * (x[i] ** 2)) - x[i - 1]) ** 2))
    return (utility + ((x[0] - 1) ** 2))

# Zakharov
def test_functionB(ind):
    utility1=0
    utility2=0
    x = ind.gene
    for i in range(N):
        utility1 = utility1 + (x[i] ** 2)
        utility2 = utility2 + (0.5 * (i + 1) * x[i])
    return (utility1 + (utility2 ** 2) + (utility2 ** 4))

# Rotated Hyper-Ellipsoid
def test_function(ind):
    utility = 0
    x = ind.gene
    for i in range(N):
        for j in range(i):
            utility += x[j] ** 2
    return utility

# INITIALISATION
def initialise(P, N, MIN, MAX):
    population = []
    for x in range(P):
        tempgene = [random.uniform(MIN, MAX) for y in range(N)]
        newind = individual()
        newind.gene = tempgene.copy()
        population.append(newind)
    return population

# EVALUATE POPULATION
def evaluate_population(population, test_function):
    for i in range(P):
        population[i].fitness = test_function(population[i])
    return population

# SELECTION
# Basic Selection
def basic_selection(population):
    offspring = []
    offspring.clear()
    for i in range(P):
        parent1 = random.randint(0, P - 1)
        off1 = copy.deepcopy(population[parent1])
        parent2 = random.randint(0, P - 1)
        off2 = copy.deepcopy(population[parent2])
        if off1.fitness < off2.fitness:
            offspring.append(off1)
        else:
            offspring.append(off2)
    return offspring

# Tournament Selection
def tournament_selection(population, T):
    offspring = []
    offspring.clear()
    for i in range(P):
        # Select T amount of random sample from population for tournament selection
        tournament = random.sample(population, T)
        # First individual in tournament is selected as best fitness
        off = tournament[0]
        # Loop through rest of tournament's individuals and find lowest fitness
        for ind in tournament[1:]:
            if ind.fitness < off.fitness:
                off = ind
        # Append offspring population with individual from tournament with lowest fitness
        offspring.append(off)
    return offspring

# CROSSOVER
# One-point Crossover
def one_crossover(offspring):
    toff1 = individual()
    toff2 = individual()
    temp = individual()
    for i in range(0, P, 2):
        toff1 = copy.deepcopy(offspring[i])
        toff2 = copy.deepcopy(offspring[i + 1])
        temp = copy.deepcopy(offspring[i])
        crosspoint = random.randint(1, N)
        for j in range(crosspoint, N):
            toff1.gene[j] = toff2.gene[j]
            toff2.gene[j] = temp.gene[j]
        offspring[i] = copy.deepcopy(toff1)
        offspring[i + 1] = copy.deepcopy(toff2)
    return offspring

#Multi-point Crossover
def multi_crossover(offspring, CROSSOVER):
    toff1 = individual()
    toff2 = individual()
    temp = individual()
    for i in range(0, P, 2):
        # Select 2 ind for crossover (and a temp copy of first ind to copy genetic material from to second ind)
        toff1 = copy.deepcopy(offspring[i])
        toff2 = copy.deepcopy(offspring[i + 1])
        temp = copy.deepcopy(offspring[i])
        # Get CROSSOVER amount of random numbers in range of N and sort to get an ascending list of random cross points (otherwise cross points would be in random order)
        crosspoints = random.sample(range(1, N), CROSSOVER)
        crosspoints.sort()
        for j in range(CROSSOVER - 1):
            # Define start and end of cross section
            cross_start = crosspoints[j]
            cross_end = crosspoints[j + 1]
            # Copy segment of genes defined by cross_start and cross_end from the second ind to the first ind
            toff1.gene[cross_start:cross_end] = toff2.gene[cross_start:cross_end]
            # Do the same to the second ind with the temp copy of the original first ind (because now first ind has already been overwritten)
            toff2.gene[cross_start:cross_end] = temp.gene[cross_start:cross_end]
        # Copy the 2 ind that has been crossed over into offspring population
        offspring[i] = copy.deepcopy(toff1)
        offspring[i + 1] = copy.deepcopy(toff2)
    return offspring

# Uniform Crossover
def uniform_crossover(offspring, CROSSOVER):
    toff1 = individual()
    toff2 = individual()
    for i in range(0, P, 2):
        # Select 2 individuals from offspring for crossover
        toff1 = copy.deepcopy(offspring[i])
        toff2 = copy.deepcopy(offspring[i + 1])
        # For every gene, there is a chance they will be crossed between the 2 individuals
        for j in range(N):
            if random.random() < CROSSOVER:
                toff1.gene[j] = toff2.gene[j]
                toff2.gene[j] = toff1.gene[j]
        # Copy the 2 ind that has been crossed over into offspring population
        offspring[i] = toff1
        offspring[i + 1] = toff2
    return offspring

# Arithmetic Crossover
def arithmetic_crossover(offspring):
    toff1 = individual()
    toff2 = individual()
    for i in range(0, P, 2):
        # Select 2 individuals from offspring for crossover
        toff1 = copy.deepcopy(offspring[i])
        toff2 = copy.deepcopy(offspring[i + 1])
        for j in range(N):
            # For every gene, create a random weight to be used in the crossover
            random_weight = random.uniform(0, 1)
            # Assign weighted average of genes to the 2 individuals
            toff1.gene[j] = toff1.gene[j] * random_weight + toff2.gene[j] * (1 - random_weight)
            toff2.gene[j] = toff1.gene[j] * (1 - random_weight) + toff2.gene[j] * random_weight
        # Copy the 2 ind that has been crossed over into offspring population
        offspring[i] = toff1
        offspring[i + 1] = toff2
    return offspring

# MUTATION
# Basic Mutation
def basic_mutation(offspring, MUTRATE, MUTSTEP, MIN, MAX):
    for i in range(P):
        newind = individual()
        newind.gene = []
        for j in range(N):
            gene = offspring[i].gene[j]
            mutprob = random.random()
            if mutprob < MUTRATE:
                alter = random.uniform(-MUTSTEP, MUTSTEP)
                gene = gene + alter
                if gene > MAX:
                    gene = MAX
                if gene < MIN:
                    gene = MIN
            newind.gene.append(gene)
        offspring[i] = copy.deepcopy(newind)
    return offspring

# Dynamic Mutation
def dynamic_mutation(offspring, MUTRATE, MUTSTEP, MIN, MAX, current_gen, max_gen):
    # Reduce MUTRATE after 50% of generations based on the current generation, then reduce it even more after 90% of generations
    if current_gen > max_gen * 0.9:
        MUTRATE -= (current_gen / max_gen)
    elif current_gen > max_gen * 0.5:
        MUTRATE -= ((current_gen / max_gen) / 10)
    else:
        MUTRATE = MUTRATE
    # Make sure MUTRATE stays within reasonable bounds
    if MUTRATE > 1.0:
        MUTRATE = 1.0
    if MUTRATE < 0.01:
        MUTRATE = 0.01
    for i in range(P):
        newind = individual()
        newind.gene = []
        for j in range(N):
            gene = offspring[i].gene[j]
            mutprob = random.random()
            if mutprob < MUTRATE:
                alter = random.gauss(0, MUTSTEP)
                gene = gene + alter
                if gene > MAX:
                    gene = MAX
                if gene < MIN:
                    gene = MIN
            newind.gene.append(gene)
        offspring[i] = copy.deepcopy(newind)
    return offspring

# Swap Mutation
def swap_mutation(offspring, MUTRATE, MUTSTEP, MIN, MAX, current_gen, max_gen):
    # Reduce MUTRATE after 50% of generations based on the current generation, then reduce it even more after 90% of generations
    if current_gen > max_gen * 0.9:
        MUTRATE -= (current_gen / max_gen)
    elif current_gen > max_gen * 0.5:
        MUTRATE -= ((current_gen / max_gen) / 10)
    else:
        MUTRATE = MUTRATE
    # Make sure MUTRATE stays within reasonable bounds
    if MUTRATE > 1.0:
        MUTRATE = 1.0
    if MUTRATE < 0.01:
        MUTRATE = 0.01
    for i in range(P):
        newind = individual()
        newind.gene = []
        for j in range(N):
            gene = offspring[i].gene[j]
            mutprob = random.random()
            if mutprob < MUTRATE:
                alter = random.gauss(0, MUTSTEP)
                gene = gene + alter
                if gene > MAX:
                    gene = MAX
                if gene < MIN:
                    gene = MIN
            newind.gene.append(gene)
        # Create random chance for gene swapping based on ratio of current generation / max generations
        chance = (current_gen / max_gen) * random.random()
        # Select 2 random genes and swap their positions (based on chance)
        if MUTRATE > chance:
            random_positions = random.sample(range(N), 2)
            value_1 = offspring[i].gene[random_positions[0]]
            value_2 = offspring[i].gene[random_positions[1]]
            newind.gene[random_positions[0]] = value_2
            newind.gene[random_positions[1]] = value_1
        offspring[i] = copy.deepcopy(newind)
    return offspring

# Inversion Mutation
def inversion_mutation(offspring, MUTRATE, MUTSTEP, MIN, MAX, current_gen, max_gen):
    # Reduce MUTRATE after 50% of generations based on the current generation, then reduce it even more after 90% of generations
    if current_gen > max_gen * 0.9:
        MUTRATE -= (current_gen / max_gen)
    elif current_gen > max_gen * 0.5:
        MUTRATE -= ((current_gen / max_gen) / 5)
    else:
        MUTRATE = MUTRATE
    # Make sure MUTRATE stays within reasonable bounds
    if MUTRATE > 1.0:
        MUTRATE = 1.0
    if MUTRATE < 0.01:
        MUTRATE = 0.01
    for i in range(P):
        newind = individual()
        newind.gene = []
        for j in range(N):
            gene = offspring[i].gene[j]
            mutprob = random.random()
            if mutprob < MUTRATE:
                alter = random.gauss(0, MUTSTEP)
                gene = gene + alter
                if gene > MAX:
                    gene = MAX
                if gene < MIN:
                    gene = MIN
            newind.gene.append(gene)
        # Create 2 crosspoints in genome and sort them in ascending order
        crosspoints = random.sample(range(1, N), 2)
        crosspoints.sort()
        # Reverse genes within this sample within the 2 crosspoints
        newind.gene[crosspoints[0]:crosspoints[1]] = reversed(newind.gene[crosspoints[0]:crosspoints[1]])
        offspring[i] = copy.deepcopy(newind)
    return offspring

# ELITISM
def elitism(population, offspring):
    # Create lists for elites and the worst individuals in offspring
    elites = []
    worst_off = []
    # Find ELITE amount of best and worst ind in population and offspring
    for elite in range(ELITE):
        bestind_pop = population[0]
        # Loop through population to find ind with best fitness
        for ind in population:
            if ind.fitness < bestind_pop.fitness:
                bestind_pop = ind
        # Add best ind to elites
        elites.append(bestind_pop)
        # Remove best ind from population (so they can't be selected for elitism multiple times)
        population.remove(bestind_pop)
        worstind_off = offspring[0]
        # Loop through offspring to find ind with worst fitness
        for ind in offspring:
            if ind.fitness > worstind_off.fitness:
                worstind_off = ind
        # Add worst ind to list
        worst_off.append(worstind_off)
        # Remove worst ind from offspring
        offspring.remove(worstind_off)
    # Extend offspring population with selected elites
    offspring.extend(elites)
    return offspring

# LOCAL SEARCH
def local_search(individual, test_function):
    # Base individual
    base_ind = copy.deepcopy(individual)
    # Higher LOCALITER values improve fitness but increase processing time
    for local in range(LOCALITER):
        gene_alter = []
        # Loop through genes and change them a bit in hopes of finding a better solution
        for gene in range(N):
            gene_alter.append(random.uniform(-0.1, 0.1))
        # Establish modified individual, initially as copy of base individual
        mod_ind = copy.deepcopy(base_ind)
        # Alter genes of modified individual
        for gene in range(N):
            mod_ind.gene[gene] += gene_alter[gene]
        # Test fitness of modified individual
        mod_ind.fitness = test_function(mod_ind)
        # If modified individual has better fitness than base individual, copy it over base individual
        if mod_ind.fitness < base_ind.fitness:
            base_ind = copy.deepcopy(mod_ind)
    return base_ind

# EVALUATE OFFSPRING
def evaluate_offspring(offspring, test_function):
    for i in range(P):
        offspring[i].fitness = test_function(offspring[i])
    return offspring

# Start timer
start = time.time()

# Result tracking lists
best_fitness_values = []
mean_fitness_values = []
fitness_gens_progress = []
mean_gens_progress = []

# Results are averaged over multiple separate runs
for run in range(RUNS):
    fitness_gens = []
    mean_gens = []
    # Initialise and evaluate population
    population = initialise(P, N, MIN, MAX)
    population = evaluate_population(population, test_function)
    # Run algorithm for a number of generations
    for gen in range(GENS):
        # Selection
        #offspring = basic_selection(population)
        offspring = tournament_selection(population, T)

        # Crossover
        #offspring = one_crossover(offspring)
        #offspring = multi_crossover(offspring, CROSSOVER)
        #offspring = uniform_crossover(offspring, CROSSOVER)
        offspring = arithmetic_crossover(offspring)

        # Mutation
        #offspring = basic_mutation(offspring, MUTRATE, MUTSTEP, MIN, MAX)
        #offspring = dynamic_mutation(offspring, MUTRATE, MUTSTEP, MIN, MAX, current_gen = gen + 1, max_gen = GENS)
        #offspring = swap_mutation(offspring, MUTRATE, MUTSTEP, MIN, MAX, current_gen = gen + 1, max_gen = GENS)
        offspring = inversion_mutation(offspring, MUTRATE, MUTSTEP, MIN, MAX, current_gen = gen + 1, max_gen = GENS)

        # Evaluate offspring
        offspring = evaluate_offspring(offspring, test_function)

        # Elitism
        offspring = elitism(population, offspring)
        
        # Local Search
        # Select pre-determined amount of individuals for local search (selecting more improves fitness but increases processing time)
        for i in range(0, P, int(P * 0.1)):
            offspring[i] = local_search(offspring[i], test_function)
        
        # Assign offspring as population of next generation
        population = copy.deepcopy(offspring)
        
        # Record best and mean fitness per generation
        bestind = population[0]
        fitness_sum = 0
        
        for ind in population:
            fitness_sum += ind.fitness
            if ind.fitness < bestind.fitness:
                bestind = ind
        
        mean_fitness = fitness_sum / P
        
        fitness_gens.append(bestind.fitness)
        mean_gens.append(mean_fitness)
    
    # Record best and mean fitness per run
    bestind = population[0]
    fitness_sum = 0
    
    for ind in population:
        fitness_sum += ind.fitness
        if ind.fitness < bestind.fitness:
            bestind = ind
    
    mean_fitness = fitness_sum / P
    
    with open(save_data, "a") as f:
        data = f"Run {run + 1}: Best Fitness: {bestind.fitness:.6f}, Mean Fitness: {mean_fitness:.6f}\n"
        print(data)
        f.write(data)
    
    best_fitness_values.append(bestind.fitness)
    mean_fitness_values.append(mean_fitness)
    fitness_gens_progress.append(fitness_gens.copy())
    mean_gens_progress.append(mean_gens.copy())

# End timer
end = time.time()

# Calculate how many seconds it took to compute
runs_duration = end - start

# Calculate averages for best and mean fitness progression over generations for pyplot
best_fitness_progress = []

for i in range(GENS):
    total = 0
    for list in fitness_gens_progress:
        total += list[i]
    average = total / len(fitness_gens_progress)
    best_fitness_progress.append(average)

mean_fitness_progress = []

for i in range(GENS):
    total = 0
    for list in mean_gens_progress:
        total += list[i]
    average = total / len(mean_gens_progress)
    mean_fitness_progress.append(average)

# Calculate average best fitness and mean fitness across all runs
total_best_fitness = 0

for best_fitness in best_fitness_values:
    total_best_fitness += best_fitness

average_best_fitness = total_best_fitness / RUNS

total_mean_fitness = 0

for mean_fitness in mean_fitness_values:
    total_mean_fitness += mean_fitness

average_mean_fitness = total_mean_fitness / RUNS

fitness_ratio = average_mean_fitness / average_best_fitness

# Results
with open(save_data, "a") as f:
    data = f"\n-- Over {RUNS} Runs: Average Best Fitness: {average_best_fitness:.6f}, Average Mean Fitness: {average_mean_fitness:.6f}, Fitness Ratio: {fitness_ratio:.3f} --\n"
    print(data)
    f.write(data)
    data = f"-- RUNS: {RUNS}, P: {P}, GENS: {GENS}, MUTRATE: {MUTRATE:.2f}, MUTSTEP: {MUTSTEP:.2f}; took {runs_duration:.2f} seconds to compute --\n"
    print(data)
    f.write(data)

# Create visualisation with pyplot
plt.plot(best_fitness_progress, label = "Best Fitness", color = "blue")
plt.plot(mean_fitness_progress, label = "Mean Fitness", color = "green")
plt.title(f"Fitness Progression\nBest Fitness: {average_best_fitness:.6f}\nMean Fitness: {average_mean_fitness:.6f}")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.tight_layout(pad = 1)
plt.legend()
plt.show()