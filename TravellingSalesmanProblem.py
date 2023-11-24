# TravellingSalesmanProblem.py

import random
from typing import List, Dict
import constants
from math import radians, sin, cos, sqrt, atan2

def haversine(coord1, coord2):
    R = 6371.0 # approximate radius of earth in km
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance

class TravellingSalesmanProblem:
    cities: List[str] = []
    distances: List = []

    @staticmethod
    def init(city_data: List[Dict[str, str]]) -> None:
        TravellingSalesmanProblem.cities = [city["name"] for city in city_data]
        TravellingSalesmanProblem.distances = []

        for i in range(len(city_data)):
            row = []
            for j in range(len(city_data)):
                lat1, lon1 = city_data[i]["latitude"], city_data[i]["longitude"]
                lat2, lon2 = city_data[j]["latitude"], city_data[j]["longitude"]
                distance = haversine((lat1, lon1), (lat2, lon2))
                row.append(distance)
            TravellingSalesmanProblem.distances.append(row)

    @staticmethod
    def get_distance(city1: int, city2: int) -> int:
        return TravellingSalesmanProblem.distances[city1][city2]

    @staticmethod
    def get_city(city_index: int) -> str:
        return TravellingSalesmanProblem.cities[city_index]

    @staticmethod
    def get_cities_index() -> List[int]:
        return [i for i in range(len(TravellingSalesmanProblem.cities))]


class Gene:
    city_index: int = None

    def __init__(self, *args, **kwargs):
        if 'gene' in kwargs:
            gene = kwargs.get('gene')
            self.city_index = gene.city_index
        elif 'city_index' in kwargs:
            city_index = kwargs.get('city_index')
            self.city_index = city_index

    def get_distance(self, gene):
        return TravellingSalesmanProblem.get_distance(self.city_index, gene.city_index)

    def __str__(self):
        return TravellingSalesmanProblem.get_city(self.city_index)

    def mutate(self) -> None:
        raise NotImplementedError("You must implement this")


class Individual:
    def __init__(self, *args, **kwargs):
        self.fitness: float = None
        self.genome: List[Gene] = []

        if len(kwargs) == 0:
            available_indexes = TravellingSalesmanProblem.get_cities_index()
            while len(available_indexes) != 0:
                index = random.randrange(len(available_indexes))
                self.genome.append(Gene(city_index=available_indexes[index]))
                del available_indexes[index]

        elif 'parent' in kwargs:
            parent = kwargs.get('parent')
            for gene in parent.genome:
                self.genome.append(Gene(gene))
            self.mutate()

        elif 'parent1' in kwargs and 'parent2' in kwargs:
            parent1 = kwargs.get('parent1')
            parent2 = kwargs.get('parent2')
            cut_point = random.randrange(len(parent1.genome))
            for i in range(cut_point):
                self.genome.append(Gene(parent1.genome[i]))
            for gene in parent2.genome:
                if gene not in self.genome:
                    self.genome.append(Gene(gene))
            self.mutate()

    def mutate(self) -> None:
        if random.random() < constants.mutation_rate:
            index1 = random.randrange(len(self.genome))
            gene = self.genome[index1]
            self.genome.remove(gene)
            index2 = random.randrange(len(self.genome))
            self.genome.insert(index2, gene)

    def evaluate(self) -> float:
        total_km = 0
        former_gene = None
        gene: Gene
        for gene in self.genome:
            if former_gene is not None:
                total_km = total_km + gene.get_distance(former_gene)
            former_gene = gene
        total_km = total_km + former_gene.get_distance(self.genome[0])
        self.fitness = total_km
        return self.fitness

    def __str__(self) -> str:
        gen = "(" + str(self.fitness) + ")"
        genome_str = ""
        for gene in self.genome:
            genome_str = genome_str + " - " + str(gene)
        return gen + " " + genome_str


class IndividualFactory:
    city_data: List[Dict[str, str]] = []

    @staticmethod
    def init(city_data: List[Dict[str, str]]) -> None:
        IndividualFactory.city_data = city_data
        TravellingSalesmanProblem.init(city_data)

    @staticmethod
    def create_individual(*args, **kwargs) -> Individual:
        if len(kwargs) == 0:
            ind = Individual()
            return ind
        elif 'parent' in kwargs:
            return Individual(kwargs.get('parent'))
        elif 'parent1' in kwargs and 'parent2' in kwargs:
            return Individual(kwargs.get('parent1'), kwargs.get('parent2'))


class EvolutionnaryProcess:
    population: List[Individual] = []
    nb_generation: int = 0
    best_fitness: float = None
    problem_name: str = ""  # Remove constants.problem_name

    def __init__(self, city_data: List[Dict[str, str]]):
        IndividualFactory.init(city_data)
        for i in range(constants.nb_individual):
            self.population.append(IndividualFactory.create_individual())

    def survival(self, new_generation: List[Individual]) -> None:
        self.population = new_generation

    def selection(self) -> Individual:
        index1: int = random.randrange(constants.nb_individual)
        index2: int = random.randrange(constants.nb_individual)
        if self.population[index1].fitness <= self.population[index2].fitness:
            return self.population[index1]
        else:
            return self.population[index2]

    def run(self):
        self.best_fitness = constants.min_fitness + 1

        while self.nb_generation < constants.nb_max_generations and self.best_fitness > constants.min_fitness:
            best_individual: Individual = self.population[0]
            for individual in self.population:
                individual.evaluate()
                if individual.fitness < best_individual.fitness:
                    best_individual = individual

            print(str(self.nb_generation) + " -> " + best_individual.__str__())
            self.best_fitness = best_individual.fitness

            new_population: List[Individual] = [best_individual]

            for i in range(constants.nb_individual):
                if random.random() < constants.crossover_rate:
                    parent1: Individual = self.selection()
                    parent2: Individual = self.selection()
                    new_population.append(IndividualFactory.create_individual(parent1, parent2))
                else:
                    parent: Individual = self.selection()
                    new_population.append(IndividualFactory.create_individual(parent))

            self.survival(new_population)
            self.nb_generation += 1

def main():
    city_data = [
        {"name": "Paris", "latitude": 48.866669, "longitude": 2.33333},
        {"name": "Lyon", "latitude": 45.764043, "longitude": 4.835659},
        {"name": "Marseille", "latitude": 43.296482, "longitude": 5.36978},
        {"name": "Nantes", "latitude": 47.218371, "longitude": -1.553621},
        {"name": "Bordeaux", "latitude": 44.837789, "longitude": -0.57918},
        {"name": "Toulouse", "latitude": 43.604652, "longitude": 1.444209},
        {"name": "Lille", "latitude": 50.62925, "longitude": 3.057256},
    ]

    syst = EvolutionnaryProcess(city_data)
    syst.run()

if __name__ == '__main__':
    main()