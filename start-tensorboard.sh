#!/bin/bash

nvidia-docker run -d --rm -ti --name=trind_population_tensorboard \
-v `pwd`:/source \
-p 2610:6006 trind/full-item /bin/bash -c "tensorboard --logdir=/source/summary"
