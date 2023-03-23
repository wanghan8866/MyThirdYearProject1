from typing import List, Tuple, Dict, Union


class SelectionType:
    def __init__(self, options: List[Union[str, int, tuple]], default=None):
        self.options = options
        self.default = default
        if len(options) == 1:
            self.default = options[0]


class NumberType:
    def __init__(self, min_: float, max_: float, default: float = None):
        self.min_ = min_
        self.max_ = max_
        self.default = default


class ListType:
    def __init__(self, current: List[int]):
        self.current = current
        self.default = current


class TrainingMeta:
    MODELS: dict = {
        "DoubleDeepQLearning": {
            "board_size": SelectionType([(10, 10)]),
            "vision_type": SelectionType([4, 8, 16], default=8),
            "hidden_network_architecture":
                ListType([64, 32]),

        },

        "Genetic": {
            #### GA stuff ####
            "board_size": SelectionType([(10, 10)]),
            'hidden_layer_activation':
                SelectionType(["relu", "sigmoid", "tanh", "linear", "leaky_relu"], default="relu"),

            "output_layer_activation":
                SelectionType(["relu", "sigmoid", "tanh", "linear", "leaky_relu"], default="sigmoid"),
            "hidden_network_architecture":
                ListType([64, 32]),
            "vision_type": SelectionType([4, 8, 16], default=8),
            ## Mutation ##
            # Mutation rate is the probability that a given gene in a chromosome will randomly mutate
            'mutation_rate': NumberType(0., 1., 0.05),  # Value must be between [0.00, 1.00)
            # If the mutation rate type is static, then the mutation rate will always be `mutation_rate`,
            # otherwise if it is decaying it will decrease as the number of generations increase
            'mutation_rate_type': SelectionType(["static", "decaying"], default="static"),
            # Options are [static, decaying]
            # The probability that if a mutation occurs, it is gaussian
            # 'probability_gaussian': NumberType(0., 1., 1.),  # Values must be between [0.00, 1.00]
            # The probability that if a mutation occurs, it is random uniform
            # 'probability_random_uniform': NumberType(0., 1., 0.),  # Values must be between [0.00, 1.00]

            ## Crossover ##

            # eta related to SBX. Larger values create a distribution closer around the parents while smaller values venture further from them.
            # Only used if probability_SBX > 0.00
            # 'SBX_eta': NumberType(0, 100, 100),
            # Probability that when crossover occurs, it is simulated binary crossover
            # 'probability_SBX': NumberType(0., 1., .5),
            # The type of SPBX to consider. If it is 'r' then it flattens a 2D array in row major ordering.
            # If SPBX_type is 'c' then it flattens a 2D array in column major ordering.
            # 'SPBX_type': 'r',  # Options are 'r' for row or 'c' for column
            # Probability that when crossover occurs, it is single point binary crossover
            # 'probability_SPBX': NumberType(0., 1., 0.5),
            # Crossover selection type determines the way in which we select individuals for crossover

            ## Selection ##

            # Number of parents that will be used for reproducing
            # 'num_parents':                 500,
            'num_parents': NumberType(1, 1000, 500),
            # Number of offspring that will be created. Keep num_offspring >= num_parents
            # 'num_offspring':               1000,
            'num_offspring': NumberType(1, 2000, 1000),
            # The selection type to use for the next generation.
            # If selection_type == 'plus':
            #     Then the top num_parents will be chosen from (num_offspring + num_parents)
            # If selection_type == 'comma':
            #     Then the top num_parents will be chosen from (num_offspring)
            # @NOTE: if the lifespan of the individual is 1, then it cannot be selected for the next generation
            # If enough indivduals are unable to be selected for the next generation, new random ones will take their place.
            # @NOTE: If selection_type == 'comma' then lifespan is ignored.
            #   This is equivalent to lifespan = 1 in this case since the parents never make it to the new generation.
            'selection_type': SelectionType(["plus", "comma"], default="plus"),  # Options are ['plus', 'comma']

            ## Individual ##

            # How long an individual is allowed to be in the population before dying off.
            # This can be useful for allowing exploration. Imagine a top individual constantly being selected for crossover.
            # With a lifespan, after a number of generations, the individual will not be selected to participate in future
            # generations. This can allow for other individuals to have higher selective pressure than they previously did.
            # @NOTE this only matters if 'selecton_type' == 'plus'. If 'selection_type' == 'comma', then 'lifespan' is completely ignored.

            # Options are any positive integer or np.inf (typed out as if it were a number, i.e. no quotes to make it a string)
            # The type of vision the snake has when it sees itself or the apple.
            # If the vision is binary, then the input into the Neural Network is 1 (can see) or 0 (cannot see).
            # If the vision is distance, then the input into the Neural Network is 1.0/distance.
            # 1.0/distance is used to keep values capped at 1.0 as a max.
            'apple_and_self_vision': SelectionType(["binary", "distance"], default="binary"),
            # Options are ['binary', 'distance']
            'generation': SelectionType(["0"], default="0"),  # Options are ['binary', 'distance'],
            "working_directory":   ListType(""),
        }

    }
