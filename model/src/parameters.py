from itertools import product

class ParameterManager:
  def __init__(self):
    
    # keeps track of our intenral list
    self.parameters = {}
    self.test_cases = []
    self.sub_test_cases = []

    # class instance iterable index
    self._index = 0


  def add_parameter(self, **kwargs):
    '''
      add a single or multiple keyword arguments to test_cases
    '''

    for key, values in kwargs.items():
      self.parameters[key] = values
      self.test_cases = [dict(zip(self.parameters.keys(), values)) for values in product(*self.parameters.values())]

  def add_nested_parameter(self, **kwargs):
    '''
      adds a nested parameter to all current test_cases
    '''
    assert len(self.test_cases) > 0, "must have a parameter to nest, use `add_parameter()` first!"

    for key, values in kwargs.items():
      self.parameters[key] = values

    # modify if using differently-sized input/output dimensions
    _input_nodes  = 784
    _output_nodes = 10

    # create all possible combinations of our neural network architectures
    for layer in self.parameters['hidden_layers']:
      activations_combos = list(product(self.parameters['activation'], repeat = layer))
      for activation_combo in activations_combos:

        # create `hidden_layers` combinations
        combo_dimensions = [(_input_nodes, self.parameters['hidden_dimensions'][0])]
        assert layer <= len(self.parameters['hidden_dimensions']), f"`hidden_dimensions` length must be more than {layer}!"

        # create a list of input/output tuples
        for i in range(layer):
          last_layer = self.parameters['hidden_dimensions'][i + 1] if i < layer - 1 else _output_nodes
          combo_dimensions.append((self.parameters['hidden_dimensions'][i], last_layer))

        # adds each 'key' combo based on our depth
        self.sub_test_cases.append({'hidden_layers': layer, 'hidden_dimensions': combo_dimensions, 'activation': activation_combo + ('softmax',)})

    # add all possible nested parameters to our copy of our test_cases
    self.copy_test_case = []
    for base in self.test_cases:
      for arch in self.sub_test_cases:
        new_case = base.copy()
        new_case.update(arch)
        self.copy_test_case.append(new_case)

    # update our test_cases
    self.test_cases = self.copy_test_case

  def length(self):
    return len(self.test_cases)

  def __iter__(self):
    '''
      wraps an iterable around this class
    '''
    self._index = 0
    return self

  def __next__(self):
    '''
      calls the next element in our iterable class's internal list object
    '''
    if self._index < len(self.test_cases):
      result = self.test_cases[self._index]
      self._index += 1
      return result
    else:
      raise StopIteration

  def __repr__(self):
    '''
      returns a debug-friendly object string representation for all test casess
    '''
    test_case_str = ''
    for index, case in enumerate(self.test_cases):
        test_case_str += f"\tcase {index + 1}: {case}\n"
    return f"\n test cases:\n{test_case_str}"