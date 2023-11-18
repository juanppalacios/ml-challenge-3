
from itertools import product

class TestRunner:
    def __init__(self, **kwargs):
        self.test_cases = kwargs
        self.test_configurations = self._generate_test_configurations()

    def _generate_test_configurations(self):
        # Generate all combinations of parameters
        parameter_combinations = product(*self.test_cases.values())

        # Collect test configurations in a list
        test_configurations = []
        for combination in parameter_combinations:
            test_config = dict(zip(self.test_cases.keys(), combination))
            test_configurations.append(test_config)

        return test_configurations

    def add_parameters(self, **kwargs):
        # Update test cases with new parameters
        self.test_cases.update(kwargs)

        if 'architecture' in kwargs:

            sub_test_cases = {
                'dimension': [],
                'function': []
            }

            architecture = kwargs['architecture']
            print(architecture)

            if 'depth' in architecture:
                depth = architecture['depth']

            for n in depth:

                if 'dimension' in architecture:
                    
                    #> input layer
                    dimension_tuples = [(architecture['dimension'][0], architecture['dimension'][1])]
                    
                    
                    #> keep adding until our output layer
                    for i in range(1, n - 1):
                        print(i)
                        last_node = architecture['dimension'][i + 1] if i < n else architecture['dimension'][n]
                        dimension_tuples.append((architecture['dimension'][i], last_node))

                    print(dimension_tuples)
                    
                    # sub_test_cases['dimension'].append(dimension_tuples)

            print(sub_test_cases)


        # Handle parameterization of 'dimension' within 'architecture'
        # if 'architecture' in kwargs:
        #     architecture_values = kwargs['architecture']
        #     if isinstance(architecture_values, list):
        #         # Update 'dimension' parameter within each architecture value
        #         for arch_value in architecture_values:
        #             if 'dimension' in arch_value:
        #                 dimension_values = arch_value['dimension']
        #                 if isinstance(dimension_values, list):
        #                     for i, dim_value in enumerate(dimension_values):
        #                         dim_key = f'dimension_{i + 1}'
        #                         arch_value[dim_key] = dim_value
        #                     del arch_value['dimension']


        # Regenerate test configurations
        self.test_configurations = self._generate_test_configurations()

    def run_tests(self):
        for config in self.test_configurations:
            print("Running test with parameters:", config)

# Example usage
test_runner = TestRunner()

# add base parameters
test_runner.add_parameters(preprocess=['none', 'normalize'])
# test_runner.add_parameters(depth=[2, 3])
test_runner.add_parameters(architecture={'depth': [4, 5], 'dimension': [784, 64, 32, 10], 'function': ['relu', 'sigmoid']})

# Run tests
# test_runner.run_tests()
