import os
import time
from algorithms import *
from algorithms import Algorithms
import threading

# folder_path = "instances/a280-ttp"
folder_path = "instances/berlin52-ttp"
file_list = os.listdir(folder_path)

# params
file_number = 0
population_size = 50
mutation_rate = 0.01
num_generations = 50
results = []
for i in range(7):
    
    min_speed = float("inf")
    max_speed = float("-inf")
    capacity_of_knapsack = 0
    cities = []
    distance_matrix = []
    items = []
    file_name = ""

    if len(file_list) > 0:
        file_name = file_list[i]
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

    arguments = [cities, distance_matrix, population_size, num_generations, mutation_rate, capacity_of_knapsack, items, max_speed, min_speed]

    Solver = Algorithms(*arguments)
    def run_algorithms_TTP(algorithms):
        for algorithm in algorithms:
            result = algorithm()
            results.append([algorithm.__name__[7:], result, file_name])
        print("Finished running TTP")

    start_time = time.time()

    for i in range(10):
        run_algorithms_TTP([Solver.runTTP_Evolutionary, Solver.runTTP_Greedy, Solver.runTTP_Random, Solver.runTTP_SA])

    end = time.time()
    print("Time:", end - start_time)

with open("results_" + file_name + ".csv", "w") as file:
    file.write("Algorithm, Fitness Score, Instance\n")
    for result in results:
        file.write(f"{result[0]}, {int(result[1])}, {result[2]}\n")
