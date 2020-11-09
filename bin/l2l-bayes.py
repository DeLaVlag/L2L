"""
This file is a typical example of a script used to run a L2L experiment. Read the comments in the file for more
explanations
"""

import logging.config
import os
import numpy as np

from l2l.utils.environment import Environment
from l2l.utils.experiment import Experiment

from l2l.logging_tools import create_shared_logger_data, configure_loggers
from l2l.optimizees.optimizee import Optimizee, OptimizeeParameters
from l2l.optimizers.optimizer import Optimizer, OptimizerParameters
from l2l.paths import Paths

from l2l.utils import JUBE_runner as jube

# from l2l.optimizees.mnist.optimizee import MNISTOptimizee
from l2l.optimizees.stan.optimizee import BayesOptimizee
from l2l.optimizers.gradientdescent.optimizer import GradientDescentOptimizer, RMSPropParameters

# We first setup the logger and read the logging config which controls the verbosity and destination of the logs from
# various parts of the code.
logger = logging.getLogger('bin.l2l-optimizee-optimizer')


def main():
    # TODO: use  the experiment module to prepare and run later the simulation
    # define a directory to store the results
    # experiment = Experiment(root_dir_path='~/home/user/L2L/results')
    experiment = Experiment(root_dir_path='/p/project/cslns/vandervlag1/L2L/bin/results')
    # TODO when using the template: use keywords to prepare the experiment and
    #  create a dictionary for jube parameters
    # prepare_experiment returns the trajectory and all jube parameters
    jube_params = {"nodes": "2",
                   "walltime": "10:00:00",
                   "ppn": "1",
                   "cpu_pp": "1"}
    traj, all_jube_params = experiment.prepare_experiment(name='L2L',
                                                          log_stdout=True,
                                                          jube_parameter=jube_params)

    ## Innerloop simulator
    # TODO when using the template: Change the optimizee to the appropriate
    #  Optimizee class
    optimizee = BayesOptimizee(traj)
    # optimizee = MNISTOptimizee(traj)
    # TODO Create optimizee parameters
    optimizee_parameters = OptimizeeParameters()

    ## Outerloop optimizer initialization
    # TODO when using the template: Change the optimizer to the appropriate
    #  Optimizer class and use the right value for optimizee_fitness_weights.
    #  Length is the number of dimensions of fitness, and negative value
    #  implies minimization and vice versa
    # optimizer_parameters = OptimizerParameters()
    # optimizer = Optimizer(traj, optimizee.create_individual, (1.0,),
    #                       optimizer_parameters)

    optimizer_parameters = RMSPropParameters(learning_rate=0.01, exploration_step_size=0.01,
                                   n_random_steps=2, momentum_decay=0.5,
                                   n_iteration=2, stop_criterion=0.0001, seed=99)

    optimizer = GradientDescentOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                         optimizee_fitness_weights=(-0.1,),
                                         parameters=optimizer_parameters,
                                         optimizee_bounding_func=optimizee.bounding_func)


    experiment.run_experiment(optimizee=optimizee,
                              optimizee_parameters=optimizee_parameters,
                              optimizer=optimizer,
                              optimizer_parameters=optimizer_parameters)

    experiment.end_experiment(optimizer)


if __name__ == '__main__':
    main()
