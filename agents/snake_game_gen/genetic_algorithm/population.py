from typing import List

import numpy as np

from .individual import Individual


class Population(object):
    def __init__(self, individuals: List[Individual]):
        self.individuals = individuals

    @property
    def num_individuals(self) -> int:
        return len(self.individuals)

    @num_individuals.setter
    def num_individuals(self, val) -> None:
        raise Exception('Cannot set the number of individuals. You must change Population.individuals instead')

    @property
    def num_genes(self) -> int:
        return self.individuals[0].chromosome.shape[1]

    @num_genes.setter
    def num_genes(self, val) -> None:
        raise Exception('Cannot set the number of genes. You must change Population.individuals instead')

    @property
    def average_fitness(self) -> float:
        return np.mean(np.array([individual.fitness for individual in self.individuals]))

    @average_fitness.setter
    def average_fitness(self, val) -> None:
        raise Exception('Cannot set average fitness. This is a read-only property.')

    @property
    def fittest_individual(self) -> Individual:
        return max(self.individuals, key=lambda individual: individual.fitness)

    @fittest_individual.setter
    def fittest_individual(self, val) -> None:
        raise Exception('Cannot set fittest individual. This is a read-only property')

    def calculate_fitness(self) -> None:
        for individual in self.individuals:
            individual.calculate_fitness()

    def get_fitness_std(self) -> float:
        return np.std(np.array([individual.fitness for individual in self.individuals]))
