"""
The program calculates the optimum location of the network of measurement sites for determining 
the spatial mean value of the measured environmental parameter in the study area. 
The calculation is performed using a genetic algorithm on the basis of a time series of spatial data stored in a NetCDF file.
"""
import numpy as np
from netCDF4 import Dataset
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import random
import math
import requests

num_measurement_stations = 6    # Define the number of measurement stations
# Genetic algorithm parameters
population_size = 100 
num_generations = 200 
mutation_rate = 0.1

file_name=r'x_ss_sst_merge-sb1k_avg_month.nc'
try:
    nc=Dataset(file_name,'r')   
except:                          # if the netCDF file is not in the current directory, it is downloaded from the repository
    url=r'https://mostwiedzy.pl/pl/open-research-data/monthly-mean-sea-surface-temperature-sst-of-the-baltic-sea,830062554854123-0/download'
    data = requests.get(url).content   # download data from the repository
    nc=Dataset(file_name, memory=data)

var=nc.variables['SST']  
time_steps=var.shape[0]
#Calculation of the average value for the entire area at each time step
avg=np.empty(time_steps, dtype=float)
for t in range(time_steps):
    avg[t]=var[t].mean()
data=np.moveaxis(var, 0, -1)    
# Define the search space boundaries
extent_min =(0, 0)
extent_max = (var.shape[1]-1,var.shape[2]-1)

# Define the cost function to be minimized 
def cost_function(stations): 
    #Initialization
    y = avg
    regr = linear_model.LinearRegression()
    x=np.empty((time_steps,len(stations)),float)
    for station_nr in range(len(stations)):
        x[:,station_nr]=data[stations[station_nr][0],stations[station_nr][1]]
    regr.fit(x, y)           # Make a match with sklearn
    y_pred = regr.predict(x) # Make predictions using the testing set
    MSE=mean_squared_error(y, y_pred)   #Calculate match stats
    return MSE,regr

# Initialize a random population of measurement stations
def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        position=[]
        while len(position)<num_measurement_stations:
            pos=list(np.random.randint(extent_min,extent_max,2))
            if data[tuple(pos)].min()!=var.missing_value:
                position.append(pos)
        population.append(position)       
    return population
	
# Select individuals based on their fitness using tournament selection
def tournament_selection(population, fitness_values, tournament_size=3):
    tournament_indices = random.sample(range(len(population)), tournament_size)
    selected_index = min(tournament_indices, key=lambda i: fitness_values[i])
    return population[selected_index]

# Apply single-point crossover to create offspring
def crossover(parent1, parent2):
    crossover_point = random.randint(1, num_measurement_stations - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Apply mutation to an individual
def mutate(individual):
    for i in range(num_measurement_stations):        
        if random.uniform(0, 1) < mutation_rate:
            while True:
                pos=list(np.random.randint(extent_min,extent_max,2))
                if data[tuple(pos)].min()!=var.missing_value:
                    individual[i]=pos
                    break
    return individual
	
# Run the genetic algorithm calculations
population = initialize_population(population_size)

for generation in range(num_generations):
    fitness_values = [cost_function(individual)[0] for individual in population]

    new_population = []
    for _ in range(population_size // 2):
        parent1 = tournament_selection(population, fitness_values)
        parent2 = tournament_selection(population, fitness_values)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1)
        child2 = mutate(child2)
        new_population.extend([child1, child2])

    best_fitness = min(fitness_values)
    best_individual = population[fitness_values.index(best_fitness)]
    print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")
    new_population.append(best_individual)
    population = new_population

cost,best_coef=cost_function(best_individual)
print("Optimal location of measurement stations: ", best_individual)
print(f"RMSE: {np.sqrt(cost):.4f}, intercept: {best_coef.intercept_:.4f}")
print("Regression coefficients:",best_coef.coef_)

