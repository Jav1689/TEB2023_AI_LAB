"""Find a short round trip using the assignment's fictional distances."""
from itertools import permutations
from random import Random

LOCATIONS = "ABCDEFG"
DISTANCE = (
    (0, 4, 8, 6, 7, 9, 5),
    (4, 0, 5, 7, 3, 8, 6),
    (8, 5, 0, 4, 6, 3, 7),
    (6, 7, 4, 0, 5, 6, 8),
    (7, 3, 6, 5, 0, 4, 6),
    (9, 8, 3, 6, 4, 0, 5),
    (5, 6, 7, 8, 6, 5, 0),
)
DEPOT = 0
DELIVERY_STOPS = tuple(range(1, len(LOCATIONS)))
SEED = 42
POPULATION_SIZE = 30
GENERATIONS = 60
MUTATION_RATE = 0.20


def route_distance(route):
    trip = (DEPOT, *route, DEPOT)
    return sum(DISTANCE[start][end] for start, end in zip(trip, trip[1:]))


def select_parent(population, rng):
    # In each tournament, the shortest of three sampled routes wins.
    return min(rng.sample(population, 3), key=route_distance)


def crossover(first, second, rng):
    # Copy a slice from the first parent, then fill the remaining slots
    # in the second parent's order, skipping stops already copied.
    start, end = sorted(rng.sample(range(len(first)), 2))
    fixed = first[start:end + 1]
    remaining = tuple(stop for stop in second if stop not in fixed)
    return remaining[:start] + fixed + remaining[start:]


def mutate(route, rng):
    if rng.random() < MUTATION_RATE:
        stops = list(route)
        left, right = rng.sample(range(len(stops)), 2)
        stops[left], stops[right] = stops[right], stops[left]
        return tuple(stops)
    return route


def run_ga():
    rng = Random(SEED)
    population = [
        tuple(rng.sample(DELIVERY_STOPS, len(DELIVERY_STOPS)))
        for _ in range(POPULATION_SIZE)
    ]
    for _ in range(GENERATIONS):
        # Elitism keeps the best route intact through selection and mutation.
        next_population = [min(population, key=route_distance)]
        while len(next_population) < POPULATION_SIZE:
            first = select_parent(population, rng)
            second = select_parent(population, rng)
            next_population.append(mutate(crossover(first, second, rng), rng))
        population = next_population
    return min(population, key=route_distance)


def main():
    best_route = run_ga()
    trip = (DEPOT, *best_route, DEPOT)
    print(f"Random seed: {SEED}")
    print(f"Generations: {GENERATIONS}")
    print("Best GA route: " + " -> ".join(LOCATIONS[stop] for stop in trip))
    print(f"Total distance: {route_distance(best_route)} km")
    # Independent check for this small instance. The GA does not use it.
    exact_distance = min(map(route_distance, permutations(DELIVERY_STOPS)))
    print(f"Exhaustive check (720 orders): {exact_distance} km")


if __name__ == "__main__":
    main()
