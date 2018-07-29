#!/bin/bash

nvidia-docker run -d --rm -ti \
-v `pwd`:/source \
-v /root/code/all_dataset:/dataset \
-p 2609:8888 trind/full-item /bin/bash -c "jupyter notebook --allow-root"
