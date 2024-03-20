import os
import time
from algorithms import *
from algorithms import GeneticAlgorithm
from algorithms import GreedyAlgorithm
from algorithms import RandomAlgorithm

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
items = []

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
                for line in file:
                    if line.startswith("ITEMS SECTION"):
                        for line in file:
                            if line.startswith("EOF"):
                                break
                            else:
                                item, profit, weight, assigned_node = line.split()
                                items.append((item, int(profit), int(weight), int(assigned_node)))
                    else:
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
EA = GeneticAlgorithm(cities, distance_matrix, population_size, num_generations, mutation_rate, capacity_of_knapsack, items)
GA = GreedyAlgorithm(cities, distance_matrix, capacity_of_knapsack, items)
RA = RandomAlgorithm(cities, distance_matrix , capacity_of_knapsack, items, num_generations, population_size)

# start_time = time.time()
# best_individualEA = EA.runTSP()
# end = time.time()
# print("Time: EA", end - start_time)

# start_time = time.time()
# best_individualGA = GA.runTSP()
# end = time.time()
# print("Time GA:", end - start_time)

# start_time = time.time()
# best_individualRA = RA.runTSP()
# end = time.time()
# print("Time RA:", end - start_time)

algorithms = [EA, GA, RA]
# for algorithm in algorithms:
#     fitness_score = calculate_fitness(algorithm.runTSP(), distance_matrix)
#     print(f"Fitness Score TSP [{algorithm.__class__.__name__}]: {fitness_score}")

# for algorithm in algorithms:
#     fitness_score = sum(item[1] for item in algorithm.runKNP())
#     print(f"Fitness Score KNP [{algorithm.__class__.__name__}]: {fitness_score}")

print(GA.runTTP())
print(RA.runTTP())