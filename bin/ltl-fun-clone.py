import os
import warnings
import logging.config
import yaml
from pypet import Environment
from pypet import pypetconstants
from ltl.optimizees.functions.optimizee import FunctionOptimizee
from ltl.optimizers.clone.optimizer import CloneOptimizer, CloneParameters
from ltl.paths import Paths
import ipdb

warnings.filterwarnings("ignore")

logger = logging.getLogger('ltl-fun-clone')


def main():
    name = 'LTL-FUN-CLONE'
    root_dir_path = 'results'  # CHANGE THIS to the directory where your simulation results are contained
    os.chdir('/home/thomas/workspace/LTL/src/')
    assert root_dir_path is not None, \
           "You have not set the root path to store your results." \
           " Set it manually in the code (by setting the variable 'root_dir_path')" \
           " before running the simulation"
    paths = Paths(name, dict(run_no='test'), root_dir_path=root_dir_path)

    with open("bin/logging.yaml") as f:
        l_dict = yaml.load(f)
        log_output_file = os.path.join(paths.results_path, l_dict['handlers']['file']['filename'])
        l_dict['handlers']['file']['filename'] = log_output_file
        logging.config.dictConfig(l_dict)

    print("All output can be found in file ", log_output_file)
    print("Change the values in logging.yaml to control log level and destination")
    print("e.g. change the handler to console for the loggers you're interesting in to get output to stdout")

    traj_file = os.path.join(paths.output_dir_path, 'data.h5')

    # Create an environment that handles running our simulation
    # This initializes a PyPet environment
    env = Environment(trajectory=name, filename=traj_file, file_title='{} data'.format(name),
                      comment='{} data'.format(name),
                      add_time=True,
                      freeze_input=True,
                      multiproc=True,
                      use_scoop=True,
                      wrap_mode=pypetconstants.WRAP_MODE_LOCAL,
                      automatic_storing=True,
                      log_stdout=False,  # Sends stdout to logs
                      log_folder=os.path.join(paths.output_dir_path, 'logs')
                      )

    # Get the trajectory from the environment
    traj = env.trajectory

    # NOTE: Innerloop simulator
    optimizee = FunctionOptimizee(traj, 'rastrigin')

    # NOTE: Outerloop optimizer initialization
    # TODO: Change the optimizer to the appropriate Optimizer class
    parameters = CloneParameters(pop_size=50, rho=0.1, n_iteration=30, burn_in=3)
    optimizer = CloneOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                            optimizee_fitness_weights=(-0.1,),
                                            parameters=parameters,
                                            optimizee_bounding_func=optimizee.bounding_func)

    # Add post processing
    env.add_postprocessing(optimizer.post_process)

    # Run the simulation with all parameter combinations
    env.run(optimizee.simulate)

    # NOTE: Innerloop optimizee end
    optimizee.end()
    # NOTE: Outerloop optimizer end
    optimizer.end()

    # Finally disable logging and close all log-files
    env.disable_logging()


if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        main()