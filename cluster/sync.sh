#!/usr/bin/bash
# note that to use delftblue here you need to have ssh keys set up and add the entry for the server in your ~/.ssh/config
rsync -trau --itemize-changes cluster/sbatches delftblue:learn_models_minatar/cluster/