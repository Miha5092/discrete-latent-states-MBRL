# in this example, I will generate 5 sbatch files for training dqn with seeds 0, 1, 2, 3, 4

import os
import pathlib

seeds = [0, 1, 2, 3, 4]
for seed in seeds:
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
        hours = 6
        minutes = 30
        template = template.replace("HOURS", str(hours))
        template = template.replace("MINUTES", str(minutes))
        # assume we need 32 GB of memory
        memory = 32*1024
        template = template.replace("MEMORY", str(memory))

        """
        command
        if you don't use container, remove ./run from the command
        """
        command = f"python3 dqn.py -o dqn_training_seed_{seed} -s -v -g quick_breakout --seed={seed}"
        template = template.replace("COMMAND", command)
        
        if not os.path.exists("cluster/sbatches"):
            os.makedirs("cluster/sbatches")
        
        """
        save template
        """
        with open(f"cluster/sbatches/dqn_training_seed_{seed}.sbatch", "w") as f:
            f.write(template)

