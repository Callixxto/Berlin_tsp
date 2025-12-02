import math
import random
import matplotlib.pyplot as plt


class TSP:
    def __init__(self):
        self.cities = {}  # City_id -> {"x": float, "y": float}
        self.load_tsp()

    #Load the .tsp file
    def load_tsp(self):                                             #Task 1
        with open("berlin52.tsp", "r") as f:
            lines = f.readlines()
            start = lines.index("NODE_COORD_SECTION\n") + 1
            fin = lines.index("EOF\n")

        for i in lines[start:fin]:
            parts = i.split()
            city_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            self.cities[city_id] = {"x": x, "y": y}

    #Distance between two cities
    def distance(self, id1: int, id2: int) -> float:
        c1 = self.cities[id1]
        c2 = self.cities[id2]
        dx = c1["x"] - c2["x"]
        dy = c1["y"] - c2["y"]
        return math.sqrt(dx * dx + dy * dy)

    def random_solution(self):
        city_ids = list(self.cities.keys())
        random.shuffle(city_ids)
        return city_ids

    def fitness(self, solution):                                #Task 5
        total = 0.0
        for i in range(len(solution) - 1):
            total += self.distance(solution[i], solution[i + 1])

        total += self.distance(solution[-1], solution[0])
        return total

    def info(self, solution):                                   #Task 6
        #Route
        print("Route:", " ".join(str(city) for city in solution))

        #Compute score
        score = self.fitness(solution)
        print("Score:", score)

    def greedy(self, start_id: int):
        unvisited = set(self.cities.keys())
        current = start_id
        tour = [current]
        unvisited.remove(current)

        while unvisited:
            closest_city = None
            closest_distance = float("inf")

            # manually search for nearest neighbour
            for cid in unvisited:
                d = self.distance(current, cid)
                if d < closest_distance:
                    closest_distance = d
                    closest_city = cid

            tour.append(closest_city)
            unvisited.remove(closest_city)
            current = closest_city

        return tour

    def initial_population(self, size, greedy_ratio=0.3):               #Task 12
        population = []

        #How many greedy and random
        num_greedy = int(size * greedy_ratio)
        num_random = size - num_greedy

        #Add greedy individuals by simply running greedy from random starting cities
        city_ids = list(self.cities.keys())

        for i in range(num_greedy):
            start = random.choice(city_ids)
            tour = self.greedy(start)
            population.append(tour)

        #Add random individuals
        for i in range(num_random):
            population.append(self.random_solution())

        return population


    def population_info(self, population):                          #Task 13
        # Compute all fitness values
        scores = [self.fitness(sol) for sol in population]

        # Basic stats
        size = len(population)
        best = min(scores)
        worst = max(scores)

        #Middle element for median for odd and even examples
        sorted_scores = sorted(scores)
        mid = size // 2
        if size % 2 == 1:
            median = sorted_scores[mid]
        else:
            median = (sorted_scores[mid - 1] + sorted_scores[mid]) / 2

        print("\~Population Info~")
        print("Population size:", size)
        print("Best score:     ", best)
        print("Median score:   ", median)
        print("Worst score:    ", worst)


    def tournament_selection(self, population, k=3):                   #Task 14
        contestants = random.sample(population, k)
        best = None
        best_score = float("inf")

        for sol in contestants:
            score = self.fitness(sol)
            if score < best_score:
                best_score = score
                best = sol

        return best

    def crossover_OC(self, parent1, parent2):                          #Task 15
        size = len(parent1)

        #Picking two cut points
        a = random.randint(0, size - 2)
        b = random.randint(a + 1, size - 1)

        #Empty child
        child = [None] * size

        #Copy part of parent 1
        for i in range(a, b + 1):
            child[i] = parent1[i]

        #Filling the rest with parent 2
        pos = 0
        for i in range(size):
            if parent2[i] not in child:
                while child[pos] is not None:                   # find next empty spot
                    pos += 1
                child[pos] = parent2[i]

        return child

    def mutate_swap(self, individual, pm=0.05):                    #Task 16
        #Swap mutation. For every city, with probability pm swap with another random city.
        mutated = individual[:]  #Copy so as not to overwrite
        size = len(mutated)

        for i in range(size):
            if random.random() < pm:
                # choose a position to swap with
                j = random.randint(0, size - 1)
                # swap the cities
                mutated[i], mutated[j] = mutated[j], mutated[i]

        return mutated

                          #prop of crossover / prop of mutation
    def epoch(self, population, pc=0.9, pm=0.05, tournament_k=3):
        new_population = []
        best_individual = None
        best_score = float("inf")

        pop_size = len(population)

        while len(new_population) < pop_size:
            #select parents
            p1 = self.tournament_selection(population, k=tournament_k)
            p2 = self.tournament_selection(population, k=tournament_k)

            #crossover with probability pc
            if random.random() < pc:
                child = self.crossover_OC(p1, p2)
            else:
                child = p1[:]  # copy parent1 (no crossover)

            #mutation (swap) with probability pm per city
            child = self.mutate_swap(child, pm)

            #evaluate (fitness)
            score = self.fitness(child)

            #add to new population
            new_population.append(child)

            #track best individual in this epoch
            if score < best_score:
                best_score = score
                best_individual = child

        return new_population, best_individual, best_score

def run_ga_once(tsp, pop_size, epochs, pc, pm, tournament_k, greedy_ratio=0.0):
    # Create initial population
    population = tsp.initial_population(pop_size, greedy_ratio=greedy_ratio)

    best_score_overall = float("inf")

    for _ in range(epochs):
        population, best_ind, best_score = tsp.epoch(
            population,
            pc=pc,
            pm=pm,
            tournament_k=tournament_k
        )
        if best_score < best_score_overall:
            best_score_overall = best_score

    return best_score_overall  # return final best score


if __name__ == "__main__":
    tsp = TSP()  #From load_tsp()

    #Run greedy for every possible starting city and print info() for each. Find best start and  save its score as reference.
    best_start = None
    best_score = None
    best_solution = None
    greedy_scores = []  # list of (start_id, score)

    for start_id in sorted(tsp.cities.keys()):
        print(f"\nGreedy starting from city {start_id}:")
        solution = tsp.greedy(start_id)
        tsp.info(solution)  # prints route and its score

        score = tsp.fitness(solution)
        greedy_scores.append((start_id, score))

        if best_score is None or score < best_score:
            best_score = score
            best_start = start_id
            best_solution = solution

    print("\nGreedy summary")
    all_greedy_scores = [s for (_, s) in greedy_scores]
    same_scores = (len(set(all_greedy_scores)) == 1)
    print("All greedy scores the same?:", same_scores)
    print("Best starting city:", best_start)
    print("Best greedy score (reference):", best_score)
    print("Best greedy route:",
          " ".join(str(city) for city in best_solution))

    random_results = []  # list of (index, score, solution)

    #Gen. 100 results
    for i in range(100):
        print(f"\nRandom solution {i + 1}:")
        sol = tsp.random_solution()
        tsp.info(sol)  # prints route + score

        s = tsp.fitness(sol)
        random_results.append((i + 1, s, sol))

    # Comparison random vs greedy
    random_scores = [s for (_, s, _) in random_results]

    print("\nRandom vs Greedy Comparison")
    print("Greedy best score:", best_score)
    print("Best random score:", min(random_scores))
    print("Worst random score:", max(random_scores))
    print("Average random score:", sum(random_scores) / len(random_scores))

    #Task 13
    population = tsp.initial_population(1023)
    tsp.population_info(population)

    #Task 14
    parent1 = tsp.tournament_selection(population, k=3)
    parent2 = tsp.tournament_selection(population, k=3)
    #Parent info
    print("\nParent 1:")
    tsp.info(parent1)
    print("\nParent 2:")
    tsp.info(parent2)

    #Task 15
    child = tsp.crossover_OC(parent1, parent2)
    print("\nChild from OC crossover:")
    tsp.info(child)

    #Task 16
    mutated_child = tsp.mutate_swap(child, pm=0.1)
    print("\nMutated Child:")
    tsp.info(mutated_child)

    #Task 17
    # create initial population (epoch 0)
    pop0 = tsp.initial_population(size=1023)
    print("Epoch 0:")
    tsp.population_info(pop0)

    # create new epoch (epoch 1)
    pop1, best_ind, best_score = tsp.epoch(pop0, pc=0.9, pm=0.05)

    print("\nEpoch 1:")
    tsp.population_info(pop1)
    print("\nBest individual in epoch 1:")
    tsp.info(best_ind)

    #Task 18

    POP_SIZE = 200  # you can tune this
    EPOCHS = 50  # number of epochs/generations
    PC = 0.9  # crossover probability
    PM = 0.01  # mutation probability

    # Initial population for GA (epoch 0)
    ga_pop = tsp.initial_population(size=POP_SIZE, greedy_ratio=0.0)
    print("\n=== GA Epoch 0 ===")
    tsp.population_info(ga_pop)

    best_overall_ind = None
    best_overall_score = float("inf")
    best_scores_per_epoch = []

    # global best starts as best in initial population
    global_best = min(tsp.fitness(sol) for sol in ga_pop)

    for epoch_idx in range(1, EPOCHS + 1):
        ga_pop, best_ind_epoch, best_score_epoch = tsp.epoch(
            ga_pop, pc=PC, pm=PM, tournament_k=4
        )

        print(f"\n=== GA Epoch {epoch_idx} ===")
        tsp.population_info(ga_pop)

        # store just this epoch's best (can go up/down)
        best_scores_per_epoch.append(best_score_epoch)

        if best_score_epoch < best_overall_score:
            best_overall_score = best_score_epoch
            best_overall_ind = best_ind_epoch

    # Recompute greedy best (for clean comparison, independent of earlier vars)
    greedy_best_score = None
    greedy_best_route = None
    for start_id in sorted(tsp.cities.keys()):
        sol = tsp.greedy(start_id)
        score = tsp.fitness(sol)
        if greedy_best_score is None or score < greedy_best_score:
            greedy_best_score = score
            greedy_best_route = sol

    # Compare GA vs greedy (and vs random, from earlier if you want)
    print("\nFINAL GA SUMMARY:")
    print("Best greedy score:", greedy_best_score)
    print("Best GA score:    ", best_overall_score)

    print("\nBest GA individual:")
    tsp.info(best_overall_ind)

    print("\nBest greedy route (for reference):")
    tsp.info(greedy_best_route)

    #Task 19

    # Plot best score as a function of epoch
    epochs = list(range(len(best_scores_per_epoch)))  # 0..EPOCHS
    plt.figure()
    plt.plot(epochs, best_scores_per_epoch, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Best score in population")
    plt.title("GA progress over epochs")

    # Optional: show greedy reference as horizontal line
    plt.axhline(greedy_best_score, color="red", linestyle="--", label="Greedy best")
    plt.legend()
    plt.show()

    #Task 20
    print("\n~PARAMETER COMPARISON~")

    TEST_EPOCHS = 40   # how long each GA run is
    POP = 200          # keep population constant for fairness

    # Parameter sets to test:
    mutation_values = [0.001, 0.01, 0.05, 0.1]
    crossover_values = [0.7, 0.9]
    tournament_values = [3, 4, 5]

    # Store results
    results_mut = []
    for pm in mutation_values:
        score = run_ga_once(tsp, POP, TEST_EPOCHS, pc=0.9, pm=pm, tournament_k=3)
        results_mut.append(score)

    results_cross = []
    for pc in crossover_values:
        score = run_ga_once(tsp, POP, TEST_EPOCHS, pc=pc, pm=0.01, tournament_k=3)
        results_cross.append(score)

    results_tourn = []
    for k in tournament_values:
        score = run_ga_once(tsp, POP, TEST_EPOCHS, pc=0.9, pm=0.01, tournament_k=k)
        results_tourn.append(score)

    #Mutation comparison
    plt.figure()
    plt.plot(mutation_values, results_mut, marker="o")
    plt.xlabel("Mutation probability")
    plt.ylabel("Best score after GA")
    plt.title("Effect of mutation rate on GA quality")
    plt.grid()
    plt.show()

    #Crossover comparison
    plt.figure()
    plt.plot(crossover_values, results_cross, marker="o")
    plt.xlabel("Crossover probability")
    plt.ylabel("Best score after GA")
    plt.title("Effect of crossover rate on GA quality")
    plt.grid()
    plt.show()

    #Tournament size comparison
    plt.figure()
    plt.plot(tournament_values, results_tourn, marker="o")
    plt.xlabel("Tournament size")
    plt.ylabel("Best score after GA")
    plt.title("Effect of tournament size on GA quality")
    plt.grid()
    plt.show()

