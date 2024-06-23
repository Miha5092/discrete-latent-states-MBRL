# in this example, I will generate 5 sbatch files for training dqn with seeds 0, 1, 2, 3, 4

import os
import pathlib

batch_sizes = [64, 128, 256, 512, 1024, 2048]
regularization = 1e-05

for batch_size in batch_sizes:
    # read template
    with open("cluster/template.sbatch", "r") as f:
        template = "".join(f.readlines())
        
        """
        configuration
        """
        # assume we need to use 1 gpu
        gpus = 1
        template = template.replace("GPUS", str(gpus))
        template = template.replace("PARTITION", "gpu-v100")
        # assume we need to use 8 cpus
        cpus = 8
        template = template.replace("CPUS", str(cpus))
        # assume we need to run for 6 hours and 30 minutes
        hours = 3
        minutes = 59
        template = template.replace("HOURS", str(hours))
        template = template.replace("MINUTES", str(minutes))
        # assume we need 32 GB of memory
        memory = 32*1024
        template = template.replace("MEMORY", str(memory))

        """
        command
        if you don't use container, remove ./run from the command
        """
        command = f"python3 learn_model.py --unroll_steps=1 --num_epochs=20 --image_prediction_loss=MSE --out_channels=128 --data_path=data/agent_training/dqn_training_seed_0/experience_data_full.npz --learning_rate=0.00025 --exp_name=test_bs_{batch_size}_reg_{regularization} --seed=0 --weight_decay={regularization} --batch_size={batch_size}"
        template = template.replace("COMMAND", command)
        
        if not os.path.exists("cluster/sbatches"):
            os.makedirs("cluster/sbatches")
        
        """
        save template
        """
        sbatch_name = f"test_bs_{batch_size}_reg_{str(regularization)}.sbatch"
        sbatch_path = pathlib.Path("cluster/sbatches") / sbatch_name
        with open(sbatch_path, "w") as f:
            f.write(template)

        """
        sync to delftblue
        """
        os.system(f"rsync -trau {sbatch_path} delftblue:learn_models_minatar/{sbatch_path}")

        """
        execute the sbatch file on delftblue
        """
        os.system(f"ssh delftblue 'cd learn_models_minatar; sbatch {sbatch_path}'")