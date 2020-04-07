import logging.config
import os

from l2l.utils.environment import Environment

from l2l.logging_tools import create_shared_logger_data, configure_loggers
from l2l.optimizees.snn.optimizee import StructuralPlasticityOptimizee, \
    StructuralPlasticityOptimizeeParameters
from l2l.optimizers.kalmanfilter import EnsembleKalmanFilter,\
    EnsembleKalmanFilterParameters
from l2l.paths import Paths
import l2l.utils.JUBE_runner as jube

logger = logging.getLogger('bin.l2l-mnist-enkf')


def run_experiment():
    name = 'L2L-SNN-ENKF'
    try:
        with open('path.conf') as f:
            root_dir_path = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            "You have not set the root path to store your results."
            " Write the path to a path.conf text file in the bin directory"
            " before running the simulation")

    trajectory_name = 'snn-enkf'

    paths = Paths(name, dict(run_num='test'), root_dir_path=root_dir_path,
                  suffix="-" + trajectory_name)

    print("All output logs can be found in directory ", paths.logs_path)

    # Create an environment that handles running our simulation
    # This initializes an environment
    env = Environment(
        trajectory=trajectory_name,
        filename=paths.output_dir_path,
        file_title='{} data'.format(name),
        comment='{} data'.format(name),
        add_time=True,
        automatic_storing=True,
        log_stdout=False,  # Sends stdout to logs
    )

    create_shared_logger_data(
        logger_names=['bin', 'optimizers'],
        log_levels=['INFO', 'INFO'],
        log_to_consoles=[True, True],
        sim_name=name,
        log_directory=paths.logs_path)
    configure_loggers()

    # Get the trajectory from the environment
    traj = env.trajectory

    # Set JUBE params
    traj.f_add_parameter_group("JUBE_params", "Contains JUBE parameters")

    # Scheduler parameters
    # Name of the scheduler
    # traj.f_add_parameter_to_group("JUBE_params", "scheduler", "Slurm")
    # Command to submit jobs to the schedulers
    traj.f_add_parameter_to_group("JUBE_params", "submit_cmd", "sbatch")
    # Template file for the particular scheduler
    traj.f_add_parameter_to_group("JUBE_params", "job_file", "job.run")
    # Number of nodes to request for each run
    traj.f_add_parameter_to_group("JUBE_params", "nodes", "1")
    # Requested time for the compute resources
    traj.f_add_parameter_to_group("JUBE_params", "walltime", "01:00:00")
    # MPI Processes per node
    traj.f_add_parameter_to_group("JUBE_params", "ppn", "1")
    # CPU cores per MPI process
    traj.f_add_parameter_to_group("JUBE_params", "cpu_pp", "1")
    # Threads per process
    traj.f_add_parameter_to_group("JUBE_params", "threads_pp", "1")
    # Type of emails to be sent from the scheduler
    traj.f_add_parameter_to_group("JUBE_params", "mail_mode", "ALL")
    # Email to notify events from the scheduler
    traj.f_add_parameter_to_group("JUBE_params", "mail_address",
                                  "a.yegenoglu@fz-juelich.de")
    # Error file for the job
    traj.f_add_parameter_to_group("JUBE_params", "err_file", "stderr")
    # Output file for the job
    traj.f_add_parameter_to_group("JUBE_params", "out_file", "stdout")
    # JUBE parameters for multiprocessing. Relevant even without scheduler.
    # MPI Processes per job
    traj.f_add_parameter_to_group("JUBE_params", "tasks_per_job", "1")
    # The execution command
    traj.f_add_parameter_to_group("JUBE_params", "exec", "mpirun python " +
                                  os.path.join(paths.root_dir_path,
                                               "run_files/run_optimizee.py"))
    # Ready file for a generation
    traj.f_add_parameter_to_group("JUBE_params", "ready_file",
                                  os.path.join(paths.root_dir_path,
                                               "ready_files/ready_w_"))
    # Path where the job will be executed
    traj.f_add_parameter_to_group("JUBE_params", "work_path",
                                  paths.root_dir_path)
    traj.f_add_parameter_to_group("JUBE_params", "paths_obj", paths)

    # Optimizee params
    optimizee_seed = 123
    optimizee_parameters = StructuralPlasticityOptimizeeParameters(
        seed=optimizee_seed, path=paths.root_dir_path)
    # Inner-loop simulator
    optimizee = StructuralPlasticityOptimizee(traj, optimizee_parameters)
    jube.prepare_optimizee(optimizee, paths.root_dir_path)

    logger.info("Optimizee parameters: %s", optimizee_parameters)

    # Outer-loop optimizer initialization
    optimizer_seed = 1234
    optimizer_parameters = EnsembleKalmanFilterParameters(gamma=0.1,
                                                          maxit=1,
                                                          n_ensembles=1,
                                                          n_iteration=2,
                                                          pop_size=1,
                                                          n_batches=1,
                                                          online=False,
                                                          seed=1234,
                                                          path=paths.root_dir_path)
    logger.info("Optimizer parameters: %s", optimizer_parameters)

    optimizer = EnsembleKalmanFilter(traj,
                                     optimizee_prepare = optimizee.connect_network,
                                     optimizee_create_individual=optimizee.create_individual,
                                     optimizee_fitness_weights=(1.,),
                                     parameters=optimizer_parameters,
                                     optimizee_bounding_func=None)

    # Add post processing
    env.add_postprocessing(optimizer.post_process)

    # Run the simulation with all parameter combinations
    env.run(optimizee.simulate)

    # Outer-loop optimizer end
    optimizer.end(traj)

    # Finally disable logging and close all log-files
    env.disable_logging()

    return traj.v_storage_service.filename, traj.v_name, paths


def main():
    filename, trajname, paths = run_experiment()
    logger.info("Plotting now")


if __name__ == '__main__':
    main()
