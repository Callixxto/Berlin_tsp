import math
import random
import matplotlib.pyplot as plt
import statistics as stats

class TSP:
    def __init__(self):
        self.cities = {}
        self.load_tsp()

    def load_tsp(self):
        with open("kroA100.tsp", "r") as f:
            lines = f.readlines()
            start = lines.index("NODE_COORD_SECTION\n") + 1
            fin = lines.index("EOF\n")

        for line in lines[start:fin]:
            parts = line.split()
            city_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            self.cities[city_id] = {"x": x, "y": y}

    def distance(self, id1: int, id2: int) -> float:
        c1 = self.cities[id1]
        c2 = self.cities[id2]
        dx = c1["x"] - c2["x"]
        dy = c1["y"] - c2["y"]
        return math.sqrt(dx * dx + dy * dy)

    #random solution
    def random_solution(self):
        city_ids = list(self.cities.keys())
        random.shuffle(city_ids)
        return city_ids

    #Task 5
    def fitness(self, solution):
        total = 0.0
        for i in range(len(solution) - 1):
            total += self.distance(solution[i], solution[i + 1])
        # return to start
        total += self.distance(solution[-1], solution[0])
        return total

    #Task 6
    def info(self, solution):
        print("Route:", " ".join(str(city) for city in solution))
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
            for cid in unvisited:
                d = self.distance(current, cid)
                if d < closest_distance:
                    closest_distance = d
                    closest_city = cid
            tour.append(closest_city)
            unvisited.remove(closest_city)
            current = closest_city
        return tour

    #Task 12
    def initial_population(self, size, greedy_ratio=0.3):
        population = []
        num_greedy = int(size * greedy_ratio)
        num_random = size - num_greedy
        city_ids = list(self.cities.keys())

        # greedy-based individuals
        for _ in range(num_greedy):
            start = random.choice(city_ids)
            tour = self.greedy(start)
            population.append(tour)
        # random individuals
        for _ in range(num_random):
            population.append(self.random_solution())
        return population

    #Task 13
    def population_info(self, population):
        scores = [self.fitness(sol) for sol in population]
        size = len(population)
        best = min(scores)
        worst = max(scores)
        sorted_scores = sorted(scores)
        mid = size // 2
        if size % 2 == 1:
            median = sorted_scores[mid]
        else:
            median = (sorted_scores[mid - 1] + sorted_scores[mid]) / 2

        print("\nPopulation Info")
        print("Population size:", size)
        print("Best score:     ", best)
        print("Median score:   ", median)
        print("Worst score:    ", worst)

    # Task 14
    def tournament_selection(self, population, k=3):
        contestants = random.sample(population, k)
        best = None
        best_score = float("inf")
        for sol in contestants:
            score = self.fitness(sol)
            if score < best_score:
                best_score = score
                best = sol
        return best

    #Task 15
    def crossover_OC(self, parent1, parent2):
        size = len(parent1)
        a = random.randint(0, size - 2)
        b = random.randint(a + 1, size - 1)
        child = [None] * size

        for i in range(a, b + 1):
            child[i] = parent1[i]

        pos = 0
        for i in range(size):
            if parent2[i] not in child:
                while child[pos] is not None:
                    pos += 1
                child[pos] = parent2[i]

        return child

    #Task 16
    def mutate_inversion(self, individual, pm=0.05):
        mutated = individual[:]
        if random.random() >= pm:
            return mutated  # no mutation
        a, b = sorted(random.sample(range(len(mutated)), 2))
        mutated[a:b + 1] = reversed(mutated[a:b + 1])
        return mutated

    #Task 17
    def epoch(self, population, pc=0.9, pm=0.05, tournament_k=3):
        pop_size = len(population)

        # find elite
        best_parent = None
        best_parent_score = float("inf")
        for ind in population:
            s = self.fitness(ind)
            if s < best_parent_score:
                best_parent_score = s
                best_parent = ind

        new_population = [best_parent[:]]  # start with elite
        best_individual = best_parent[:]
        best_score = best_parent_score

        while len(new_population) < pop_size:
            p1 = self.tournament_selection(population, k=tournament_k)
            p2 = self.tournament_selection(population, k=tournament_k)

            if random.random() < pc:
                child = self.crossover_OC(p1, p2)
            else:
                child = p1[:]

            child = self.mutate_inversion(child, pm)
            score = self.fitness(child)
            new_population.append(child)

            if score < best_score:
                best_score = score
                best_individual = child

        return new_population, best_individual, best_score

    def plot_tour(self, tour, title="Tour"):
        xs = [self.cities[c]["x"] for c in tour]
        ys = [self.cities[c]["y"] for c in tour]
        # close loop
        xs.append(self.cities[tour[0]]["x"])
        ys.append(self.cities[tour[0]]["y"])

        plt.figure(figsize=(6, 6))
        plt.plot(xs, ys, marker="o")

        for cid in tour:
            x = self.cities[cid]["x"]
            y = self.cities[cid]["y"]
            plt.text(x, y, str(cid), fontsize=8, ha="right", va="bottom")
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.show()

#Task 20: run GA once with given params
def run_ga_once(tsp, pop_size, epochs, pc, pm, tournament_k,
                greedy_ratio=0.0, record_history=False, seed=None):

    if seed is not None:
        random.seed(seed)

    # initial population
    population = tsp.initial_population(pop_size, greedy_ratio=greedy_ratio)

    best_score_overall = float("inf")
    history = []

    # best in initial population (epoch 0)
    for ind in population:
        s = tsp.fitness(ind)
        if s < best_score_overall:
            best_score_overall = s
    if record_history:
        history.append(best_score_overall)

    # GA epochs
    for _ in range(epochs):
        population, best_ind, best_score = tsp.epoch(
            population,
            pc=pc,
            pm=pm,
            tournament_k=tournament_k
        )
        if best_score < best_score_overall:
            best_score_overall = best_score
        if record_history:
            history.append(best_score_overall)

    if record_history:
        return best_score_overall, history
    else:
        return best_score_overall


if __name__ == "__main__":
    tsp = TSP()

    #Task 9 (greedy baseline)
    best_start = None
    best_greedy_score = None
    best_greedy_route = None
    greedy_scores = []

    for start_id in sorted(tsp.cities.keys()):
        print(f"\nGreedy starting from city {start_id}:")
        route = tsp.greedy(start_id)
        tsp.info(route)

        score = tsp.fitness(route)
        greedy_scores.append((start_id, score))

        if best_greedy_score is None or score < best_greedy_score:
            best_greedy_score = score
            best_start = start_id
            best_greedy_route = route

    print("\nGreedy summary")
    print("Best starting city:", best_start)
    print("Best greedy score (reference):", best_greedy_score)

    #Task 10
    random_results = []
    for i in range(100):
        print(f"\nRandom solution {i + 1}:")
        sol = tsp.random_solution()
        tsp.info(sol)
        s = tsp.fitness(sol)
        random_results.append((i + 1, s, sol))

    random_scores = [s for (_, s, _) in random_results]
    print("\nRandom vs Greedy Comparison")
    print("Greedy best score:", best_greedy_score)
    print("Best random score:", min(random_scores))
    print("Worst random score:", max(random_scores))
    print("Average random score:", sum(random_scores) / len(random_scores))

    #Tasks 13–17
    demo_pop = tsp.initial_population(size=100)
    tsp.population_info(demo_pop)

    parent1 = tsp.tournament_selection(demo_pop, k=3)
    parent2 = tsp.tournament_selection(demo_pop, k=3)
    print("\nParent 1:")
    tsp.info(parent1)
    print("\nParent 2:")
    tsp.info(parent2)

    child = tsp.crossover_OC(parent1, parent2)
    print("\nChild from OC crossover:")
    tsp.info(child)

    mutated_child = tsp.mutate_inversion(child, pm=0.1)
    print("\nMutated Child:")
    tsp.info(mutated_child)

    pop0 = tsp.initial_population(size=100)
    print("\nEpoch 0 (demo):")
    tsp.population_info(pop0)
    pop1, best_ind_demo, best_score_demo = tsp.epoch(pop0, pc=0.9, pm=0.05)
    print("\nEpoch 1 (demo):")
    tsp.population_info(pop1)
    print("\nBest individual in demo epoch 1:")
    tsp.info(best_ind_demo)

    #Task 18
    GA_POP_SIZE     = 400
    GA_EPOCHS       = 900
    GA_PC           = 0.9
    GA_PM           = 0.1
    GA_TOURNAMENT_K = 10

    ga_pop = tsp.initial_population(size=GA_POP_SIZE, greedy_ratio=0.0)
    print("\nGA Epoch 0")
    tsp.population_info(ga_pop)

    best_overall_ind = None
    best_overall_score = float("inf")
    best_scores_per_epoch = []

    # best-so-far in initial population (epoch 0)
    for sol in ga_pop:
        s = tsp.fitness(sol)
        if s < best_overall_score:
            best_overall_score = s
            best_overall_ind = sol
    best_scores_per_epoch.append(best_overall_score)

    # GA loop
    for epoch_idx in range(1, GA_EPOCHS + 1):
        ga_pop, best_ind_epoch, best_score_epoch = tsp.epoch(
            ga_pop, pc=GA_PC, pm=GA_PM, tournament_k=GA_TOURNAMENT_K
        )

        print(f"\nGA Epoch {epoch_idx}")
        tsp.population_info(ga_pop)

        # update GLOBAL best (best so far across ALL epochs)
        if best_score_epoch < best_overall_score:
            best_overall_score = best_score_epoch
            best_overall_ind = best_ind_epoch

        # for the plot: store BEST SO FAR (monotone decreasing)
        best_scores_per_epoch.append(best_overall_score)

    print("\nFINAL GA SUMMARY:")
    print("Best greedy score:", best_greedy_score)
    print("Best GA score:    ", best_overall_score)

    print("\nBest GA individual:")
    tsp.info(best_overall_ind)

    print("\nBest greedy route (for reference):")
    tsp.info(best_greedy_route)

    #Task 19 – (plotting)
    best_random_score = min(random_scores)  

    epochs = list(range(len(best_scores_per_epoch)))  # 0..GA_EPOCHS
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, best_scores_per_epoch, marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Best score (best so far)")
    plt.title("GA progress over epochs")

    subtitle = (
        f"Best Greedy: {best_greedy_score:.2f}   |   "
        f"Best Random: {best_random_score:.2f}   |   "
        f"Best GA: {best_overall_score:.2f}"
    )
    plt.suptitle(subtitle, y=0.96, fontsize=9)

    plt.axhline(best_greedy_score, color="red", linestyle="--", label="Greedy best")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot tours for visualization
    tsp.plot_tour(best_greedy_route, title="Best Greedy Tour")
    tsp.plot_tour(best_overall_ind, title="Best GA Tour")

    #Task 20
    print("\nPARAMETER COMPARISON")

    TEST_EPOCHS = 200
    POP = 200
    BASE_SEED = 123

    mutation_values   = [0.001, 0.01, 0.05, 0.1,0.4]
    tournament_values = [3, 4, 5, 10]
    pop_sizes = [50, 100, 200, 400,600]
    # Mutation
    plt.figure()
    for pm in mutation_values:
        final_score, hist = run_ga_once(
            tsp, POP, TEST_EPOCHS,
            pc=0.9, pm=pm, tournament_k=3,
            greedy_ratio=0.0,
            record_history=True,
            seed=BASE_SEED
        )
        plt.plot(range(len(hist)), hist, label=f"pm={pm:.3f}")
    plt.xlabel("Generation")
    plt.ylabel("Best score (best so far)")
    plt.title("Effect of mutation rate on GA quality")
    plt.legend()
    plt.grid()
    plt.show()
    # Tournament
    plt.figure()
    for k in tournament_values:
        final_score, hist = run_ga_once(
            tsp, POP, TEST_EPOCHS,
            pc=0.9, pm=0.03, tournament_k=k,
            greedy_ratio=0.0,
            record_history=True,
            seed=BASE_SEED
        )
        plt.plot(range(len(hist)), hist, label=f"k={k}")
    plt.xlabel("Generation")
    plt.ylabel("Best score (best so far)")
    plt.title("Effect of tournament size on GA quality")
    plt.legend()
    plt.grid()
    plt.show()
    #Population
    plt.figure()
    for pop_size in pop_sizes:
        final_score, hist = run_ga_once(
            tsp,
            pop_size,
            TEST_EPOCHS,
            pc=0.9,
            pm=0.03,
            tournament_k=4,
            greedy_ratio=0.0,
            record_history=True,
            seed=BASE_SEED
        )
        plt.plot(range(len(hist)), hist, label=f"POP={pop_size}")

    plt.xlabel("Generation")
    plt.ylabel("Best score (best so far)")
    plt.title("Effect of initial population size on GA quality")
    plt.legend()
    plt.grid()
    plt.show()

    #PART 3
    #GA with best parameters
    GA_BEST_POP        = 600     
    GA_BEST_EPOCHS     = 600     
    GA_BEST_PC         = 0.9      
    GA_BEST_PM         = 0.1     
    GA_BEST_TOURN_K    = 10     
    GA_BEST_GREEDY_RATIO = 0.3

    ga_scores = []

    for run in range(1, 11):
        best_score = run_ga_once(
            tsp,
            pop_size=GA_BEST_POP,
            epochs=GA_BEST_EPOCHS,
            pc=GA_BEST_PC,
            pm=GA_BEST_PM,
            tournament_k=GA_BEST_TOURN_K,
            greedy_ratio=GA_BEST_GREEDY_RATIO,
            record_history=False,
            seed=None         # different random seed each run
        )
        ga_scores.append(best_score)
        print(f"GA run {run:2d}: best score = {best_score:.2f}")

    ga_best = min(ga_scores)
    ga_worst = max(ga_scores)
    ga_mean = stats.mean(ga_scores)
    ga_var  = stats.pvariance(ga_scores)
    ga_std  = stats.pstdev(ga_scores)

    print("\nGA (10 runs) statistics:")
    print("Scores:", [f"{s:.2f}" for s in ga_scores])
    print(f"Best   : {ga_best:.2f}")
    print(f"Worst  : {ga_worst:.2f}")
    print(f"Mean   : {ga_mean:.2f}")
    print(f"Std dev: {ga_std:.2f}")
    print(f"Var    : {ga_var:.2f}")

    #Greedy statistics
    greedy_only_scores = [score for (_, score) in greedy_scores]
    greedy_sorted = sorted(greedy_only_scores)
    best_5_greedy = greedy_sorted[:5]

    greedy_mean = stats.mean(greedy_only_scores)
    greedy_var  = stats.pvariance(greedy_only_scores)
    greedy_std  = stats.pstdev(greedy_only_scores)

    print("\nGreedy statistics (all starting cities):")
    print(f"Number of greedy runs: {len(greedy_only_scores)}")
    print("Best 5 greedy scores:", [f"{s:.2f}" for s in best_5_greedy])
    print(f"Greedy best   : {min(greedy_only_scores):.2f}")
    print(f"Greedy worst  : {max(greedy_only_scores):.2f}")
    print(f"Greedy mean   : {greedy_mean:.2f}")
    print(f"Greedy std dev: {greedy_std:.2f}")
    print(f"Greedy var    : {greedy_var:.2f}")

    #Random: 1000 independent tours
    random_1000_scores = []
    for i in range(1000):
        sol = tsp.random_solution()
        s = tsp.fitness(sol)
        random_1000_scores.append(s)

    rand_best = min(random_1000_scores)
    rand_worst = max(random_1000_scores)
    rand_mean = stats.mean(random_1000_scores)
    rand_var  = stats.pvariance(random_1000_scores)
    rand_std  = stats.pstdev(random_1000_scores)

    print("\nRandom (1000 tours) statistics:")
    print(f"Best   : {rand_best:.2f}")
    print(f"Worst  : {rand_worst:.2f}")
    print(f"Mean   : {rand_mean:.2f}")
    print(f"Std dev: {rand_std:.2f}")
    print(f"Var    : {rand_var:.2f}")

    #Chart
    print("\n============== FINAL COMPARISON SUMMARY ==============")
    print(f"{'Method':<15} {'Best':>12} {'Mean':>12} {'Std dev':>12} {'Var':>12}")
    print("-" * 65)
    print(f"{'GA (10 runs)':<15} {ga_best:12.2f} {ga_mean:12.2f} {ga_std:12.2f} {ga_var:12.2f}")
    print(f"{'Greedy':<15} {min(greedy_only_scores):12.2f} {greedy_mean:12.2f} {greedy_std:12.2f} {greedy_var:12.2f}")
    print(f"{'Random (1000)':<15} {rand_best:12.2f} {rand_mean:12.2f} {rand_std:12.2f} {rand_var:12.2f}")
