import math
import random

class TSP:
    def __init__(self, path: str):
        self.cities = {}      #City_id -> {"x": float, "y": float}
        self._load_tsp(path)

    #Load the .tsp file
    def _load_tsp(self, path: str):
        reading_coords = False
        with open(path, "r") as f:
            for line in f:
                line = line.strip()

                if line in ("", "EOF"):
                    continue

                if line == "NODE_COORD_SECTION":
                    reading_coords = True
                    continue

                if reading_coords:
                    self.parse_city_line(line)

    def parse_city_line(self, line: str):
        parts = line.split()
        if len(parts) < 3:
            return

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

    #Random solution
    def random_solution(self) -> dict:
        city_ids = list(self.cities.keys())
        random.shuffle(city_ids)
        solution = {i: cid for i, cid in enumerate(city_ids)}

        #Checking all cities for no repetition
        assert len(solution) == len(self.cities)
        assert set(solution.values()) == set(self.cities.keys())
        return solution

    def fitness(self, solution: dict) -> float:
        #Solution: position -> city_id
        order = [solution[i] for i in sorted(solution.keys())]
        total = 0.0
        for i in range(len(order) - 1):
            total += self.distance(order[i], order[i + 1])
        #Return to start
        total += self.distance(order[-1], order[0])
        return total

    def info(self, solution: dict) -> None:
        order = [solution[i] for i in sorted(solution.keys())]
        score = self.fitness(solution)
        print("Route:", " ".join(str(cid) for cid in order))
        print("Score (total distance):", score)


    def greedy(self, start_id: int) -> dict:
        unvisited = set(self.cities.keys())
        current = start_id
        order_ids = [current]
        unvisited.remove(current)

        while unvisited:
            #Choose closest city
            next_city = min(unvisited, key=lambda cid: self.distance(current, cid))
            order_ids.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        #Converting to dictionary
        return {i: cid for i, cid in enumerate(order_ids)}

if __name__ == "__main__":
    """""
    tsp = TSP("berlin11_modified.tsp") 

    print("Loaded cities:", len(tsp.cities))
    print("Example city:", tsp.cities[2])

    d = tsp.distance(1, 2)
    print(f"Distance between city 1 and 2: {d:.3f}")

    sol = tsp.random_solution()
    print("Random solution:", sol)

    best_start = None
    best_score = None
    best_solution = None

    for start_id in sorted(tsp.cities.keys()):
        print(f"\nGreedy starting from city {start_id}:")
        solution = tsp.greedy(start_id)
        tsp.info(solution)

        score = tsp.fitness(solution)
        if best_score is None or score < best_score:
            best_score = score
            best_start = start_id
            best_solution = solution

    print("\nBest starting city:", best_start)
    print("Best score (reference):", best_score)
    print("Best solution:", best_solution)

    #100 results from random solutions
    random_results = []  #List of tuples to hold (index, score, solution)

    for i in range(100):
        print(f"\nRandom solution {i + 1}:")
        sol = tsp.random_solution()
        tsp.info(sol)  #Route + score

        s = tsp.fitness(sol)
        random_results.append((i + 1, s, sol))

    #Compare random scores with greedy best
    random_scores = [s for (_, s, _) in random_results]

    print("\n----- Comparison -----")
    print("Greedy best score:", best_score)
    print("Best random score:", min(random_scores))
    print("Worst random score:", max(random_scores))
    print("Average random score:", sum(random_scores) / len(random_scores))
   #print(random_results)
    """""
    tsp = TSP("berlin52.tsp")
    best_start = None
    best_score = None
    best_solution = None

    for start_id in sorted(tsp.cities.keys()):
        print(f"\nGreedy starting from city {start_id}:")
        solution = tsp.greedy(start_id)
        tsp.info(solution)

        score = tsp.fitness(solution)
        if best_score is None or score < best_score:
            best_score = score
            best_start = start_id
            best_solution = solution

    print("\nBest starting city:", best_start)
    print("Best score (reference):", best_score)
    print("Best solution:", best_solution)

    #100 results from random solutions
    random_results = []  #List of tuples to hold (index, score, solution)

    for i in range(100):
        print(f"\nRandom solution {i + 1}:")
        sol = tsp.random_solution()
        tsp.info(sol)  #Route + score

        s = tsp.fitness(sol)
        random_results.append((i + 1, s, sol))

    #Compare random scores with best greedy
    random_scores = [s for (_, s, _) in random_results]

    print("\n ~Comparison~")
    print("Greedy best score:", best_score)
    print("Best random score:", min(random_scores))
    print("Worst random score:", max(random_scores))
    print("Average random score:", sum(random_scores) / len(random_scores))
    #print(random_results)
