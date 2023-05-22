import numpy as np
import pandas as pd

# Function to calculate the total cost of facility allocation
def calculate_total_cost(facilities, customers, distances):
    total_cost = 0
    for c in range(len(customers)):
        min_cost = float('inf')
        for f in range(len(facilities)):
            cost = distances[f][c] * facilities[f]
            if cost < min_cost:
                min_cost = cost
        total_cost += min_cost
    return total_cost

# Golden Eagle Optimizer (GEO) algorithm
def geo_algorithm(facilities, customers, distances, max_iterations, num_eagles, pa, pc, pm):
    num_facilities = len(facilities)
    num_customers = len(customers)

    # Initialize eagle positions randomly
    eagles = np.random.rand(num_eagles, num_facilities)

    for i in range(max_iterations):
        for e in range(num_eagles):
            # Randomly select two eagles
            rand_eagles = np.random.choice(num_eagles, 2, replace=False)

            # Perform crossover operation
            new_eagle = pa * eagles[e] + (1 - pa) * eagles[rand_eagles[0]] + pc * (eagles[rand_eagles[1]] - eagles[rand_eagles[0]])

            # Perform mutation operation
            if np.random.rand() < pm:
                mutation_index = np.random.randint(num_facilities)
                new_eagle[mutation_index] = np.random.rand()

            # Evaluate the new eagle's cost
            new_cost = calculate_total_cost(new_eagle, customers, distances)

            # Replace the current eagle if the new cost is better
            if new_cost < calculate_total_cost(eagles[e], customers, distances):
                eagles[e] = new_eagle

    # Find the best eagle (solution)
    best_eagle_index = np.argmin([calculate_total_cost(eagle, customers, distances) for eagle in eagles])
    best_eagle = eagles[best_eagle_index]

    return best_eagle

# Generate random data for the example
num_rows = 100
facilities = np.random.randint(1, 10, size=num_rows)
customers = np.random.randint(1, 10, size=num_rows)
distances = np.random.randint(1, 10, size=(num_rows, num_rows))

# Save the data to an Excel file
data = pd.DataFrame({'Facilities': facilities, 'Customers': customers})
for i in range(num_rows):
    data['Distance{}'.format(i)] = distances[:, i]
data.to_excel('book1.xlsx', index=False)

# Read the data from the Excel file
df = pd.read_excel('book1.xlsx')

facilities = df['Facilities'].tolist() # Capacities of the facilities
customers = df['Customers'].tolist() # Demands of the customers
distances = df.iloc[:, 2:].values
max_iterations = 100 # Maximum number of iterations
num_eagles = 50 # Number of eagles (solutions) in the population
pa = 0.8 # Probability of accepting an attribute from the first parent
pc = 0.2 # Crossover probability
pm = 0.1 # Mutation probability
best_solution = geo_algorithm(facilities, customers, distances, max_iterations, num_eagles, pa, pc, pm)
print("Best solution:", best_solution)
print("Total cost:", calculate_total_cost(best_solution, customers, distances))
