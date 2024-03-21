import os
import time
from algorithms import *
from algorithms import Algorithms
import threading

# folder_path = "instances/a280-ttp"
folder_path = "instances/berlin52-ttp"
file_list = os.listdir(folder_path)

# params
file_number = 1
population_size = 50
mutation_rate = 0.01
num_generations = 100

min_speed = float("inf")
max_speed = float("-inf")
capacity_of_knapsack = 0
cities = []
distance_matrix = []
items = []
file_name = ""

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

arguments = [cities, distance_matrix, population_size, num_generations, mutation_rate, capacity_of_knapsack, items, max_speed, min_speed]

Solver = Algorithms(*arguments)
results = []
# ...

def run_algorithms_TTP(problem, algorithms):
    for algorithm in algorithms:
        result = algorithm()
        results.append([problem, algorithm.__name__[7:], result])
    print("Finished running TTP")

def run_algorithms_TSP(problem, algorithms):
    for algorithm in algorithms:
        result = algorithm()
        results.append([problem, algorithm.__name__[7:], result])
    print("Finished running TSP")

def run_algorithms_KNP(problem, algorithms):
    for algorithm in algorithms:
        result = algorithm()
        results.append([problem, algorithm.__name__[7:], result])
    print("Finished running KNP")


# ...

start_time = time.time()

for i in range(10):
    threads = []
    threads.append(threading.Thread(target=run_algorithms_TTP, args=("TTP", [Solver.runTTP_Evolutionary, Solver.runTTP_Greedy, Solver.runTTP_Random])))
    threads.append(threading.Thread(target=run_algorithms_TSP, args=("TSP", [Solver.runTSP_Evolutionary, Solver.runTSP_Greedy, Solver.runTSP_Random])))
    threads.append(threading.Thread(target=run_algorithms_KNP, args=("KNP", [Solver.runKNP_Evolutionary, Solver.runKNP_Greedy, Solver.runKNP_Random])))
    
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    print("All threads finished", i)

end = time.time()
print("Time:", end - start_time)

# create a csv file with the results
with open("results_" + file_name + ".csv", "w") as file:
    file.write("Problem, Algorithm, Fitness Score\n")
    for result in results:
        file.write(f"{result[0]}, {result[1]}, {int(result[2])}\n")

# for i in range(10):
#     print(f"Running iteration {i+1}")
#     results.append(["TTP","Evolutionary", Solver.runTTP_Evolutionary()])
#     results.append(["TTP","Greedy", Solver.runTTP_Greedy()])
#     results.append(["TTP","Random", Solver.runTTP_Random()])
#     results.append(["TSP","Evolutionary", Solver.runTSP_Evolutionary()])
#     results.append(["TSP","Greedy", Solver.runTSP_Greedy()])
#     results.append(["TSP","Random", Solver.runTSP_Random()])
#     results.append(["KNP","Evolutionary", Solver.runKNP_Evolutionary()])
#     results.append(["KNP","Greedy", Solver.runKNP_Greedy()])
#     results.append(["KNP","Random", Solver.runKNP_Random()])

# # create a csv file with the results
# with open("results.csv", "w") as file:
#     file.write("Problem, Algorithm, Fitness Score\n")
#     for result in results:
#         file.write(f"{result[0]}, {result[1]}, {result[2]}\n")




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

# algorithms = [EA, GA, RA]
# for algorithm in algorithms:
#     fitness_score = calculate_fitness(algorithm.runTSP(), distance_matrix)
#     print(f"Fitness Score TSP [{algorithm.__class__.__name__}]: {fitness_score}")

# for algorithm in algorithms:
#     fitness_score = sum(item[1] for item in algorithm.runKNP())
#     print(f"Fitness Score KNP [{algorithm.__class__.__name__}]: {fitness_score}")