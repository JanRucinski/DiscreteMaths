import random

# genetic algorithm class
class GeneticAlgorithm:
    cities = []
    distance_matrix = []
    items = []
    population_size = 0
    num_generations = 0
    mutation_rate = 0
    capacity_of_knapsack = 0

    def __init__(self, cities, distance_matrix, population_size, num_generations, mutation_rate, capacity_of_knapsack = 0, items = []):
        self.cities = cities
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.capacity_of_knapsack = capacity_of_knapsack

    def runTSP(self):
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
    
    def runKNP(self):
        # tbd
        return self.items
    
# greedy algorithm class
class GreedyAlgorithm:
    cities = []
    distance_matrix = []
    capacity_of_knapsack = 0
    items = []

    def __init__(self, cities, distance_matrix, capacity_of_knapsack = 0, items = []):
        self.cities = cities
        self.distance_matrix = distance_matrix
        self.capacity_of_knapsack = capacity_of_knapsack
        self.items = items

    def runTSP(self):
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
    
    def runKNP(self):
        # Sort items by profit/weight ratio
        sorted_items = sorted(self.items, key=lambda item: item[1]/item[2], reverse=True)
        knapsack = []
        knapsack_weight = 0
        knapsack_value = 0

        # Add items to knapsack
        for item in sorted_items:
            if knapsack_weight + item[2] <= self.capacity_of_knapsack:
                knapsack.append(item)
                knapsack_weight += item[2]
                knapsack_value += item[1]
        return knapsack

# random algorithm class
class RandomAlgorithm:
    cities = []
    distance_matrix = []
    capacity_of_knapsack = 0
    items = []
    num_generations = 0
    population_size = 0

    def __init__(self, cities, distance_matrix , capacity_of_knapsack = 0, items = [], num_generations = 1, population_size = 1):
        self.cities = cities
        self.distance_matrix = distance_matrix
        self.capacity_of_knapsack = capacity_of_knapsack
        self.items = items
        self.num_generations = num_generations
        self.population_size = population_size

    def runTSP(self):
        solutions = []
        while len(solutions) < self.num_generations * self.population_size:
            solution = random.sample(self.cities, len(self.cities))
            solutions.append(solution)
        return max(solutions, key=lambda solution: calculate_fitness(solution, self.distance_matrix))

    def is_knapsack_valid(self, knapsack, capacity_of_knapsack):
        total_weight = sum(item[2] for item in knapsack)
        return total_weight <= capacity_of_knapsack

    def runKNP(self):
        solutions = []
        while len(solutions) < self.num_generations * self.population_size:
            solution = random.sample(self.items, len(self.items))
            while not (self.is_knapsack_valid(solution, self.capacity_of_knapsack)):
                solution = random.sample(self.items, len(self.items))
            solutions.append(solution)
        return max(solutions, key=lambda solution: sum(item[1] for item in solution))

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

def selection(population, fitness_scores):
    # Select parents using tournament selection
    tournament_size = 5
    parents = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        winner = min(tournament, key=lambda x: x[1])[0]
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