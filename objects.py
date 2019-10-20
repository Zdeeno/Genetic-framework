import numpy as np


class Population:

    def __init__(self, population, fitness_f, selection_f, operators_f, replacement_f=None):
        """
        Class representing whole population
        :param init_f: function to initialize the population array in form [chromosome_size, population_size]
        :param fitness_f: function to calculate fitness of single chromosome
        :param operators_f: array of operators to perturbate single chromosome
        :param selection_f: selection of new population
        """
        assert population.ndim == 2
        self.population = population
        self.children_population = None
        self._fitness_f = fitness_f
        self._operators_f = operators_f
        self._selection_f = selection_f
        self._replacement_f = replacement_f
        self.fitness = np.zeros((np.shape(self.population)[1]))
        self.children_fitness = None

    def evaluate_population(self, args=()):
        self.fitness = self._fitness_f(self.population, *args)

    def evaluate_children(self, args=()):
        self.children_fitness = self._fitness_f(self.children_population, *args)

    def run_operators(self, args=None):
        assert self.children_population is not None
        for idx, operator in enumerate(self._operators_f):
            if args is not None and args[idx] is not None:
                self.children_population = operator(self.children_population, *args[idx])
            else:
                self.children_population = operator(self.children_population)

    def select_parents(self, args=()):
        self.children_population = self._selection_f(self.population, self.fitness, *args)

    def do_replacement(self, args=()):
        assert self.children_population is not None
        if self._replacement_f is not None:
            self.population = self._replacement_f(self.population, self.children_population, *args)
            self.children_population = None
        else:
            self.population = self.children_population
            self.children_population = None

