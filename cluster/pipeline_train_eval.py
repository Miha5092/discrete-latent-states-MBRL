import os, time


def submit_job(sbatch_file):
    # Submit the sbatch file and capture the output
    cmd = f'sbatch {sbatch_file}'
    stream = os.popen(cmd)
    output = stream.read()

    if stream.close() is not None:
        print(f"Error submitting job: {output}")
        return None

    # Extract the job ID from the output
    job_id = output.split()[-1]  # Assuming the job ID is the last word in the output
    return job_id


def submit_dependent_job(dependent_sbatch_file, dependency_job_id):
    # Submit the dependent job with the dependency flag
    cmd = f'sbatch --dependency=afterok:{dependency_job_id} {dependent_sbatch_file}'
    stream = os.popen(cmd)
    output = stream.read()

    if stream.close() is not None:
        print(f"Error submitting dependent job: {output}")
        return None

    # Extract the job ID from the output
    dependent_job_id = output.split()[-1]  # Assuming the job ID is the last word in the output
    return dependent_job_id


def submit_pair(main_job, dependent_job):

    # Submit the first job
    job_id = submit_job(main_job)

    time.sleep(0.15)

    if job_id:
        print(f"Submitted job with ID: {job_id}")

        # Submit the dependent job
        dependent_job_id = submit_dependent_job(dependent_job, job_id)
        if dependent_job_id:
            print(f"Submitted dependent job with ID: {dependent_job_id}")
        else:
            print("Failed to submit dependent job.")
    else:
        print("Failed to submit job.")


def generate_training(
        unroll_steps: int = 1,
        batch_size: int = 128,
        num_epochs: int = 50,
        learning_rate: float = 0.00025,
        hidden_channels: int = 128,
        seed: int = 42,
        exp_name: str = "test",
        data_path: str = "data/dqn_training_seed_0/experience_data_episode_10000.npz",
        image_prediction_loss_clip: float = 0.0,
        weight_decay: float = 0,
        latent_state_type: str = "deterministic",
        latent_state_shape: tuple = (128, ),
        kl_clip: float = 1.0,
        dyn_loss_scale: float = 0.5,
        rep_loss_scale: float = 0.1,
        kl_beta: float = 0,
        aggregate_method: str = "sum",
        monitor_model: bool = False,
        save_intermediary: bool = False
):
    name = exp_name

    hours = 1
    minutes = 59
    cpus = 1
    gpus = 1
    memory = 4 * 1024
    partition = "gpu-v100,gpu-a100"

    with open("cluster/conda_gpu_template.sbatch", "r") as f:
        template = "".join(f.readlines())

        template = template.replace("NAME", str(name))
        template = template.replace("HOURS", str(hours))
        template = template.replace("MINUTES", str(minutes))
        template = template.replace("CPUS", str(cpus))
        template = template.replace("GPUS", str(gpus))
        template = template.replace("MEMORY", str(memory))
        template = template.replace("PARTITION", str(partition))

        command = (
            f"python3 learn_model.py "
            f"--unroll_steps={unroll_steps} "
            f"--batch_size={batch_size} "
            f"--num_epochs={num_epochs} "
            f"--learning_rate={learning_rate} "
            f"--hidden_channels={hidden_channels} "
            f"--seed={seed} "
            f"--exp_name=\"{exp_name}\" "
            f"--data_path=\"{data_path}\" "
            f"--image_prediction_loss_clip={image_prediction_loss_clip} "
            f"--weight_decay={weight_decay} "
            f"--latent_state_type=\"{latent_state_type}\" "
            f"--latent_state_shape=\"{latent_state_shape}\" "
            f"--kl_clip={kl_clip} "
            f"--dyn_loss_scale={dyn_loss_scale} "
            f"--rep_loss_scale={rep_loss_scale} "
            f"--kl_beta={kl_beta} "
            f"--aggregate_method=\"{aggregate_method}\" "
            f"--monitor_model={monitor_model} "
            f"--save_intermediary={save_intermediary}"
        )

        template = template.replace("COMMAND", command)

        save_location = "cluster/sbatches/train"
        if not os.path.exists(save_location):
            os.makedirs(save_location)

        save_location = save_location + "/" + name
        with open(save_location, "w") as f:
            f.write(template)

    return save_location

def generate_eval(
        exp_name: str,
        use_mpi: bool,
        seed: int = 42,
        num_episodes: int = 100,
        n_tasks: int = 32
        ):
    name = exp_name
    partition = "compute-p1,compute-p2"

    if use_mpi:
        hours = 2
        minutes = 0
        cpus = 1
        memory_per_cpu = int(1.5 * 1024)

        with open("cluster/conda_cpu_mpi_template.sbatch", "r") as f:
            template = "".join(f.readlines())

            template = template.replace("NAME", str(name))
            template = template.replace("HOURS", str(hours))
            template = template.replace("MINUTES", str(minutes))
            template = template.replace("TASKS", str(n_tasks))
            template = template.replace("CPUS", str(cpus))
            template = template.replace("MEMORY", str(memory_per_cpu))
            template = template.replace("PARTITION", str(partition))

        command = (
            f"python3 evaluate_mpi.py "
            f"--output_folder=\"data/model_evaluation/{exp_name}\" "
            f"--model_folder=\"data/model_learning/{exp_name}\" "
            f"--seed={seed} "
            f"--num_episodes={num_episodes} "
        )

    else:
        hours = 6
        minutes = 0
        cpus = 32
        memory = 8 * 1024

        with open("cluster/conda_cpu_template.sbatch", "r") as f:
            template = "".join(f.readlines())

            template = template.replace("NAME", str(name))
            template = template.replace("HOURS", str(hours))
            template = template.replace("MINUTES", str(minutes))
            template = template.replace("CPUS", str(cpus))
            template = template.replace("MEMORY", str(memory))
            template = template.replace("PARTITION", str(partition))

        command = (
            f"python3 evaluate_model.py "
            f"--output_folder=\"data/model_evaluation/{exp_name}\" "
            f"--model_folder=\"data/model_learning/{exp_name}\" "
            f"--seed={seed} "
            f"--num_episodes={num_episodes} "
        )

    template = template.replace("COMMAND", command)

    save_location = "cluster/sbatches/eval"
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    save_location = save_location + "/" + name
    with open(save_location, "w") as f:
        f.write(template)

    return save_location

def main():
    # model_types = ["deterministic", "gaussian", "discrete"]
    # latent_shapes = [(128, ), (128, ), (32, 32)]

    epochs = 50

    model_types = ["discrete"]
    latent_shapes = [(16, 16)]

    dyn_weights = [0.7, 0.8, 0.9, 1.0]
    rep_weights = [0.05, 0.1, 0.15, 0.2]
    kl_clips = [0.75, 1.0, 1.25, 1.5]

    for model_index, model_type in enumerate(model_types):
        for dyn_weight in dyn_weights:
            for rep_weight in rep_weights:
                for clip in kl_clips:

                    exp_name = f"{model_type}_dyn_{dyn_weight}_rep_{rep_weight}_clip_{clip}"

                    training_job = generate_training(
                        exp_name=exp_name,
                        dyn_loss_scale=dyn_weight,
                        rep_loss_scale=rep_weight,
                        latent_state_type=model_type,
                        latent_state_shape=latent_shapes[model_index],
                        kl_clip=clip,
                        num_epochs=epochs
                    )

                    eval_job = generate_eval(
                        exp_name=exp_name,
                        use_mpi=True,
                        seed=42,
                        num_episodes=100
                    )

                    submit_pair(training_job, eval_job)


if __name__ == "__main__":
    main()
