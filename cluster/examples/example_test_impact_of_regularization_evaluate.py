# in this example, I will generate 5 sbatch files for training dqn with seeds 0, 1, 2, 3, 4

import os
import pathlib

regularizations = [0.0, 0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-06]
epochs = list(range(20))
for regularization in regularizations:
    for epoch in epochs:
        # read template
        with open("cluster/template.sbatch", "r") as f:
            template = "".join(f.readlines())
            
            """
            configuration
            """
            # assume we need to use 1 gpu
            gpus = 0
            template = template.replace("GPUS", str(gpus))
            template = template.replace("PARTITION", "compute-p1,compute-p2")
            # assume we need to use 8 cpus
            cpus = 4
            template = template.replace("CPUS", str(cpus))
            # assume we need to run for 6 hours and 30 minutes
            hours = 4
            minutes = 0
            template = template.replace("HOURS", str(hours))
            template = template.replace("MINUTES", str(minutes))
            # assume we need 32 GB of memory
            memory = 16*1024
            template = template.replace("MEMORY", str(memory))

            """
            command
            if you don't use container, remove ./run from the command
            """
            command = f"python3 evaluate_model.py --output_folder=data/model_evaluation/test_reg_{str(regularization)}_{str(epoch)} --num_episodes=100 --seed=0 --model_folder=data/model_learning/test_reg_{str(regularization)}/epoch_{epoch}"
            template = template.replace("COMMAND", command)
            
            if not os.path.exists("cluster/sbatches"):
                os.makedirs("cluster/sbatches")
            
            """
            save template
            """
            sbatch_name = f"test_reg_{str(regularization)}_{epoch}.sbatch"
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

