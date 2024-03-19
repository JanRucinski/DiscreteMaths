import os
import random
import time
from algorithms import *
from algorithms import GeneticAlgorithm


folder_path = "instances/a280-ttp"
file_list = os.listdir(folder_path)

# params
file_number = 0
population_size = 50
mutation_rate = 0.01
num_generations = 100

min_speed = float("inf")
max_speed = float("-inf")
capacity_of_knapsack = 0
cities = []
distance_matrix = []

if len(file_list) > 0:
    file_name = file_list[file_number]
    file_path = os.path.join(folder_path, file_name)
    print("Reading file:", file_path)

    with open(file_path, "r") as file:
        for line in file:
            # if line starts with MIN SPEED: or MAX SPEED: then we want to extract the value
            if line.startswith("MIN SPEED:"):
                min_speed = float(line.split(":")[1])
            elif line.startswith("MAX SPEED:"):
                max_speed = float(line.split(":")[1])
            elif line.startswith("CAPACITY OF KNAPSACK:"):
                capacity_of_knapsack = int(line.split(":")[1])
            elif line.startswith("NODE_COORD_SECTION"):
                # read the next lines until "DEMAND_SECTION" is found
                for line in file:
                    if line.startswith("DEMAND_SECTION") or line.startswith("ITEMS SECTION") or line.startswith("DEPOT_SECTION") or line.startswith("DEPOT_SECTION"):
                        break
                    city, x, y = line.split()
                    cities.append((city, (float(x), float(y))))
else:
    print("No files found in the folder.")

def distance(city1, city2):
    return ((city1[1][0] - city2[1][0]) ** 2 + (city1[1][1] - city2[1][1]) ** 2) ** 0.5

for i in range(len(cities)):
    row = []
    for j in range(len(cities)):
        row.append(distance(cities[i], cities[j]))
    distance_matrix.append(row)

arguments = [cities, distance_matrix, population_size, num_generations, mutation_rate]

# Run the genetic algorithm
EA = GeneticAlgorithm(*arguments)

start_time = time.time()
best_individual = EA.run()
end = time.time()
print("Time:", end - start_time)


# Print the best individual and its fitness score
# print_individual(best_individual)
# print("Best Individual:", best_individual)
print("Fitness Score:", calculate_fitness(best_individual, distance_matrix))

