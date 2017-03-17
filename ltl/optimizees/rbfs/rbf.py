import numpy as np


class RBF:
    def __init__(self, rbf_params, dims):
        cost_functions = {}
        self.bound = [-5, 5]
        name_list = ['gaussian', 'permutation']
        function_list = [Gaussian, Permutation]

        # Create a dictionary which associate the function and state bound to a cost name
        for n, f in zip(name_list, function_list):
            cost_functions[n] = f

        self.gen_functions = []
        for param in rbf_params:
            f_name = param["name"]
            function_class = cost_functions[f_name](param["params"], dims)
            self.gen_functions.append(function_class.call)

    def cost_function(self, x):
        res = 0.
        for f in self.gen_functions:
            res += f(x)

        return res


class Permutation:
    def __init__(self, params, dims):
        if not len(params) == 1:
            raise Exception("Number of parameters does not equal 1.")
        beta = np.array(params[0])

        if np.isscalar(beta):
            raise Exception("Beta paramater must always be a scalar value.")
        self.dims = dims
        self.beta = beta

    def call(self, x):
        # sum_{k=1}^n (sum_{i=1}^n [i^k+beta][(x_i/i)^k-1])^2
        x = np.array(x)
        ks = np.array(range(1, self.dims+1))
        i = np.array(range(1, self.dims+1))
        value = [np.sum((i**k + self.beta) * (x / i)**(k - 1)) for k in ks]
        value = np.sum(value**2)
        return value


class Gaussian:
    def __init__(self, params, dims):
        if not len(params) == 2:
            raise Exception("Number of parameters does not equal 2.")
        sigma = np.array(params[0])
        mean = np.array(params[1])

        if (dims > 1 and (not sigma.shape[0] == sigma.shape[1] == mean.shape[0] == dims)) or \
                (dims == 1 and (not sigma.shape == mean.shape == tuple())):
            raise Exception("Shapes do not match the given dimensionality.")
        self.dims = dims
        self.sigma = sigma
        self.mean = mean

    def call(self, x):
        x = np.array(x)
        value = 1 / np.sqrt((2*np.pi)**self.dims * np.linalg.det(self.sigma))
        value = value * np.exp(-0.5 * (np.transpose(x - self.mean).dot(np.linalg.inv(self.sigma))).dot((x - self.mean)))
        return value
