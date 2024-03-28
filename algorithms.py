import datetime
import random
import statistics

# genetic algorithm class
class Algorithms:
    cities = []
    distance_matrix = []
    items = []
    population_size = 0
    num_generations = 0
    mutation_rate = 0
    capacity_of_knapsack = 0
    Vmax = 1
    Vmin = 0.1

    def __init__(self, cities, distance_matrix, population_size, 
                 num_generations, mutation_rate, capacity_of_knapsack = 0, 
                 items = [], Vmax = 1, Vmin = 0.1):
        self.cities = cities
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.capacity_of_knapsack = capacity_of_knapsack
        self.items = items
        self.Vmax = Vmax
        self.Vmin = Vmin

    # evolutionary --------------------------------
    def runTSP_Evolutionary(self):
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
        best_individual = min(population, key=fitness)

        return self.calculate_fitness(best_individual)
    
    def runKNP_Evolutionary(self):
        num_generations = self.num_generations
        mutation_rate = self.mutation_rate
        def fitness(individual):
            total_value = sum(item[1] for item in individual)
            return total_value

        # Generate initial population
        population = generate_population_knapsack(self.population_size, 
                                                  self.items, self.capacity_of_knapsack)

        for generation in range(num_generations):
            # Evaluate fitness of each individual in the population
            fitness_scores = [fitness(individual) for individual in population]

            # Select parents for reproduction
            parents = selection_KNP(population, fitness_scores)

            # Create offspring through crossover
            offspring = crossover_knapsack(parents, self.capacity_of_knapsack)

            # Mutate offspring
            mutated_offspring = mutation(offspring, mutation_rate)

            # Replace population with offspring
            population = mutated_offspring

        # Select the best individual as the solution
        best_individual = max(population, key=lambda individual: 
                              sum(item[1] for item in individual))
        if not (is_knapsack_valid(best_individual, self.capacity_of_knapsack)):
            print("Invalid knapsack")
        
        return self.calculate_fitness_knapsack(best_individual)
    
    def runTTP_Evolutionary(self):
        generations_summary = []
        num_generations = self.num_generations
        mutation_rate = self.mutation_rate
        population_size = self.population_size
        def fitness(individual):
            total_profit = 0
            current_city = individual[0][3]
            knapsack_weight = individual[0][2]

            for item in individual[1:]:
                if knapsack_weight + item[2] <= self.capacity_of_knapsack:
                    speed = calculate_speed(self.Vmax, self.Vmin, knapsack_weight, self.capacity_of_knapsack)
                    distance = self.distance_matrix[current_city-1][item[3]-1]
                    total_profit += item[1] - (distance / speed)
                    current_city = item[3]
                    knapsack_weight += item[2]
            return total_profit

        # Generate initial population
        population = generate_population_knapsack(population_size, 
                                                  self.items, self.capacity_of_knapsack)
        counter = 0
        for generation in range(num_generations):
            # Evaluate fitness of each individual in the population
            fitness_scores = [fitness(individual) for individual in population]
            generations_summary.append([counter, max(fitness_scores), min(fitness_scores), sum(fitness_scores)/len(fitness_scores)])
            counter += 1
            # Select parents for reproduction
            parents = selection_TTP(population, fitness_scores)

            # Create offspring through crossover
            offspring = crossover_knapsack(parents, self.capacity_of_knapsack)

            # Mutate offspring
            mutated_offspring = mutation(offspring, mutation_rate)

            # Replace population with offspring
            population = mutated_offspring
        
        # Select the best individual as the solution
        best_individual = max(population, key=fitness)

        # write generations_summary to a file
        name = (f"{num_generations}, {population_size}, {mutation_rate}")
        with open("generations/" + name +".csv", "w") as file:
            file.write("Generation Max Min Average\n")
            for line in generations_summary:
                file.write(f"{line[0]} {line[1]} {line[2]} {line[3]}\n")

        return fitness(best_individual)
        
    # greedy --------------------------------
    def runTSP_Greedy(self):
        results = []
        for _ in range(self.population_size * self.num_generations):
            results.append(self.runTSP_Greedy_once())
        return min(results)
    
    def runTSP_Greedy_once(self):
        # Start with a random city
        current_city = random.choice(self.cities)
        unvisited_cities = self.cities.copy()
        unvisited_cities.remove(current_city)
        path = [current_city]

        # Visit each city by closest distance
        while unvisited_cities:
            next_city = min(unvisited_cities, key=lambda city: 
                            self.distance_matrix[int(current_city[0])-1][int(city[0])-1])
            path.append(next_city)
            unvisited_cities.remove(next_city)
            current_city = next_city
        return self.calculate_fitness(path)
    
    def runKNP_Greedy(self):
        results = []
        for _ in range(self.population_size * self.num_generations):
            results.append(self.runKNP_Greedy_once())
        return max(results)
    
    def runKNP_Greedy_once(self):
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
        return self.calculate_fitness_knapsack(knapsack)
    
    def runTTP_Greedy(self):
        results = []
        for _ in range(self.population_size * self.num_generations):
            results.append(self.runTTP_Greedy_once())
        return max(results)
    
    def runTTP_Greedy_once(self):
        # Sort items by profit/weight ratio
        sorted_items = sorted(self.items, key=lambda item: item[1]/item[2], reverse=True)
        # travel from city to city, collect items and put them in the knapsack,
        # profit is the sum of the values of the items in the knapsack
        # minus the cost of the travel (distance / speed)
        current_city = sorted_items[0][3]
        profit = sorted_items[0][1]
        knapsack_weight = sorted_items[0][2]

        for item in sorted_items[1:]:
            if knapsack_weight + item[2] <= self.capacity_of_knapsack:
                speed = calculate_speed(self.Vmax, self.Vmin, knapsack_weight, self.capacity_of_knapsack)
                distance = self.distance_matrix[current_city-1][item[3]-1]
                profit += item[1] - (distance / speed)
                current_city = item[3]
                knapsack_weight += item[2]

        return profit

    # random --------------------------------
    def runTSP_Random(self):
        solutions = generate_population(self.population_size * 
                                        self.num_generations, self.cities)
        best_solution = min(solutions, key=lambda solution:
                            self.calculate_fitness(solution))
        return self.calculate_fitness(best_solution)

    def runKNP_Random(self):
        solutions = generate_population_knapsack(self.population_size * self.num_generations, 
                                                 self.items, self.capacity_of_knapsack)
        best_solution = max(solutions, key=lambda solution: 
                            sum(item[1] for item in solution))
        return self.calculate_fitness_knapsack(best_solution)

    def runTTP_Random(self):
        solutions = []
        while len(solutions) < self.population_size * self.num_generations:
            current_city = random.choice(self.cities)
            unvisited_cities = self.cities.copy()
            unvisited_cities.remove(current_city)
            profit = self.items[(int(current_city[0])-1)][1]
            knapsack_weight = self.items[(int(current_city[0]))-1][2]
            while unvisited_cities and knapsack_weight < self.capacity_of_knapsack:
                next_city = random.choice(unvisited_cities)
                speed = calculate_speed(self.Vmax, self.Vmin, knapsack_weight, self.capacity_of_knapsack)
                distance = self.distance_matrix[(int(current_city[0]))-1][(int(next_city[0]))-1]
                profit += self.items[(int(next_city[0]))-1][1] - (distance / speed)
                current_city = next_city
                knapsack_weight += self.items[(int(next_city[0]))-1][2]
                unvisited_cities.remove(next_city)
            solutions.append(profit)
        best_solution = max(solutions)
        return best_solution
    
    # Simulated Annealing --------------------------------
    def runTTP_SA(self): #TBD
        def fitness(individual):
            total_profit = 0
            current_city = individual[0][3]
            knapsack_weight = individual[0][2]

            for item in individual[1:]:
                if knapsack_weight + item[2] <= self.capacity_of_knapsack:
                    speed = calculate_speed(self.Vmax, self.Vmin, knapsack_weight, self.capacity_of_knapsack)
                    distance = self.distance_matrix[current_city-1][item[3]-1]
                    total_profit += item[1] - (distance / speed)
                    current_city = item[3]
                    knapsack_weight += item[2]
            return total_profit

        def simulated_annealing(individual):
            temperature = 100
            cooling_rate = 0.003
            current_solution = individual
            best_solution = current_solution
            while temperature > 1:
                new_solution = mutation([current_solution], 1)[0]
                current_fitness = fitness(current_solution)
                new_fitness = fitness(new_solution)
                if new_fitness > current_fitness:
                    current_solution = new_solution
                else:
                    if random.random() < (1 / (1 + (new_fitness - current_fitness) / temperature)):
                        current_solution = new_solution
                if fitness(current_solution) > fitness(best_solution):
                    best_solution = current_solution
                temperature *= 1 - cooling_rate
            return best_solution

        population = generate_population_knapsack(self.population_size, 
                                                  self.items, self.capacity_of_knapsack)
        best_individual = simulated_annealing(population[0])
        return fitness(best_individual)

    def calculate_fitness(self, individual):
        distance_matrix = self.distance_matrix
        total_distance = 0
        for i in range(len(individual)-1):
            city1 = individual[i]
            city2 = individual[i+1]
            total_distance += distance_matrix[int(city1[0])-1][int(city2[0])-1]
        return total_distance

    def calculate_fitness_knapsack(self, individual):
        total_value = sum(item[1] for item in individual)
        return total_value

def generate_population(population_size, genes):
    population = []
    for _ in range(population_size):
        individual = random.sample(genes, len(genes))
        population.append(individual)
    return population

def generate_population_knapsack(population_size, items, capacity_of_knapsack):
    population = []
    for _ in range(population_size):
        individual = random.sample(items, len(items))
        while not is_knapsack_valid(individual, capacity_of_knapsack):
            individual = random.sample(items, len(items))
            individual = individual[:random.randint(0, len(individual))]
        population.append(individual)
    return population

def is_knapsack_valid(knapsack, capacity_of_knapsack):
    total_weight = sum(item[2] for item in knapsack)
    return total_weight <= capacity_of_knapsack and total_weight > 0

def selection(population, fitness_scores):
    # Select parents using tournament selection
    tournament_size = 5
    parents = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitness_scores)), 
                                   tournament_size)
        winner = min(tournament, key=lambda x: x[1])[0]
        parents.append(winner)
    return parents

def selection_KNP(population, fitness_scores): #TBD
    # Select parents using tournament selection
    tournament_size = 5
    parents = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitness_scores)), 
                                   tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        parents.append(winner)
    return parents


def selection_TTP(population, fitness_scores): #TBD
    # Select parents using tournament selection
    tournament_size = 5
    parents = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitness_scores)), 
                                   tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        parents.append(winner)
    return parents

def crossover(parents):
    # Perform crossover
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

def crossover_knapsack(parents, capacity_of_knapsack):
    # Perform crossover
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
        if not is_knapsack_valid(child1, capacity_of_knapsack):
            child1 = parent1
        if not is_knapsack_valid(child2, capacity_of_knapsack):
            child2 = parent2
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

def calculate_speed(Vmax, Vmin, knapsack_weight, capacity_of_knapsack):
    speed = Vmax - (((Vmax - Vmin) / capacity_of_knapsack) * knapsack_weight)
    return speed