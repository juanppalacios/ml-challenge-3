
from itertools import product

# note: whatever I wrote here, let's keep it this way...

# from toolkit import Toolkit, logging_levels


# tools = Toolkit()
# tools.configure(name = 'Parameter Manager', level = logging_levels['DEBUG'])


class ParameterManager:
  def __init__(self):
    self.parameters = {}
    self.test_cases = []

    # self.tools = Toolkit()
    # self.tools.configure(name = 'Parameter Manager', level = logging_levels['DEBUG'])

  def add_parameter(self, key = None, value = None):
    # if group is None:
    self.parameters[key] = value
    self.test_cases = list(product(*self.parameters.values()))
    # else:
      # self.parameters[key] = value
      # self.test_cases = list(product(*self.parameters.values()))

  def create_architecture(self, *args):

    # Step 1: Define layer configurations
    # preprocessing = ['none', 'normailize']
    num_layers = [2, 3]
    activation_functions_hidden = ['relu', 'sigmoid']  # Define possible activation functions for hidden layers

    # Define possible total node values for hidden layers as a list
    hidden_layer_nodes = [64, 32, 16, 8, 4]

    input_nodes = 28*28  # Fixed input layer with 28 nodes
    output_nodes = 10  # Fixed output layer with 10 nodes

    # Step 2: Use itertools.product to generate combinations
    encoded_combinations = []
    for num_hidden_layers in num_layers:
        activation_combinations = list(product(activation_functions_hidden, repeat=num_hidden_layers))

        for activation_combo in activation_combinations:
            # Create a list of (input, output) node tuples for hidden layers, including input and output layers
            #> input node
            hidden_layer_tuples = [(input_nodes, hidden_layer_nodes[0])]

            for i in range(num_hidden_layers):
                final_node = hidden_layer_nodes[i + 1] if i < num_hidden_layers - 1 else output_nodes
                hidden_layer_tuples.append((hidden_layer_nodes[i], final_node))

            encoded_combination = [num_hidden_layers, hidden_layer_tuples, activation_combo + ('softmax',)]

            encoded_combinations.append(encoded_combination)

    # Now, encoded_combinations contains all possible layer configurations with variable numbers of hidden layers
    # for combination in encoded_combinations:
    #   print(combination)
    return encoded_combinations

  def all(self):
    return self.test_cases

  def __repr__(self):
    args = '\n'
    for index, case in enumerate(self.test_cases):
      args += f'\ttest case {index}: {case}\n'
    # for key in self.parameters.keys():
      #  args += f'\t internal dictionary: {key} -> {self.parameters[key]}\n'
    return args

#> hyper-parameters
# parameters = ParameterManager()

# parameters.add_parameter('preprocessing', ['normalize'])
# test = parameters.create_architecture({
#     'layers':[4, 5]},{
#     'dimension': [64, 32, 16]},{
#     'activation': ['relu', 'sigmood']}
#     )
# parameters.add_parameter('architectures', test)

# parameters.add_parameter('epochs', [1000])
# parameters.add_parameter('learning rates', [0.01])
# parameters.add_parameter('optimizer', ['adam'])

# tools.debug(f'showing notebook parameters: {parameters}')