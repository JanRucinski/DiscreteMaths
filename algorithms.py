import random

# genetic algorithm class
class GeneticAlgorithm:
    cities = []
    distance_matrix = []
    population_size = 0
    num_generations = 0
    mutation_rate = 0

    def __init__(self, cities, distance_matrix, population_size, num_generations, mutation_rate):
        self.cities = cities
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate

    def run(self):
        num_generations = self.num_generations
        mutation_rate = self.mutation_rate
        
        def fitness(individual):
            total_distance = 0
            for i in range(len(individual)-1):
                city1 = individual[i]
                city2 = individual[i+1]
                total_distance += self.distance_matrix[int(city1[0])-1][int(city2[0])-1]
            return total_distance
        
        # Generate initial population
        population = generate_population(self.population_size, self.cities)

        for generation in range(num_generations):
            # Evaluate fitness of each individual in the population
            fitness_scores = [fitness(individual) for individual in population]

            # Select parents for reproduction
            parents = selection(population, fitness_scores)

            # Create offspring through crossover
            offspring = crossover(parents)

            # Mutate offspring
            mutated_offspring = mutation(offspring, mutation_rate)

            # Replace population with offspring
            population = mutated_offspring

        # Select the best individual as the solution
        best_individual = max(population, key=fitness)

        return best_individual
    
# greedy algorithm class
class GreedyAlgorithm:
    cities = []
    distance_matrix = []

    def __init__(self, cities, distance_matrix):
        self.cities = cities
        self.distance_matrix = distance_matrix

    def run(self):
        # Start with a random city
        current_city = random.choice(self.cities)
        unvisited_cities = self.cities.copy()
        unvisited_cities.remove(current_city)
        path = [current_city]

        # Visit each city
        while unvisited_cities:
            next_city = min(unvisited_cities, key=lambda city: self.distance_matrix[int(current_city[0])-1][int(city[0])-1])
            path.append(next_city)
            unvisited_cities.remove(next_city)
            current_city = next_city

        return path

# random algorithm class
class RandomAlgorithm:
    cities = []

    def __init__(self, cities):
        self.cities = cities

    def run(self):
        return random.sample(self.cities, len(self.cities))

def calculate_fitness(individual, distance_matrix):
    total_distance = 0
    for i in range(len(individual)-1):
        city1 = individual[i]
        city2 = individual[i+1]
        total_distance += distance_matrix[int(city1[0])-1][int(city2[0])-1]
    return total_distance

def generate_population(population_size, cities):
    population = []
    for _ in range(population_size):
        individual = random.sample(cities, len(cities))
        population.append(individual)
    return population

def fitnessCalculate(individual, distance_matrix):
    total_distance = 0
    for i in range(len(individual)-1):
        city1 = individual[i]
        city2 = individual[i+1]
        total_distance += distance_matrix[int(city1[0])-1][int(city2[0])-1]
    return total_distance

def selection(population, fitness_scores):
    # Select parents using tournament selection
    tournament_size = 5
    parents = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        parents.append(winner)
    return parents

def crossover(parents):
    # Perform random crossover
    offspring = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i+1]
        child1 = parent1[:len(parent1)//2]
        child2 = parent2[:len(parent2)//2]
        for gene in parent2:
            if gene not in child1:
                child1.append(gene)
        for gene in parent1:
            if gene not in child2:
                child2.append(gene)
        offspring.append(child1)
        offspring.append(child2)
    return offspring

def mutation(offspring, mutation_rate):
    # Perform swap mutation
    for individual in offspring:
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                j = random.randint(0, len(individual)-1)
                individual[i], individual[j] = individual[j], individual[i]
    return offspring

def print_individual(individual):
    for city in individual:
        print(city[0], end=" ")
    print()