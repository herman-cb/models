import os
from collections import defaultdict
import tensorflow as tf
import numpy as np
import pickle as pkl

def process_density():
    """
    Convert data from Tensorflow even format to a numpy dictionary serialized
    in pickel format
    """
    target_dir = "/cb/data/herman/github3376/vgg_19/"
    print("Inside process density")

    events = sorted([file for file in os.listdir(target_dir) if file.startswith('events')])

    density_stats = defaultdict(dict)
    # stores the density masks as a dict of dicts
    # first key is layer name
    # second key is iteration number

    for event in events:
        event_file = os.path.join(target_dir, event)
        for e in tf.train.summary_iterator(event_file):
            #import ipdb
            #ipdb.set_trace()
            for v in e.summary.value:
                #if v.tag.startswith("training_stats"):
                if len(v.tensor.tensor_shape.dim) > 0:
                #if len(v.tensor.tensor_shape)
                    shapes = [x.size for x in v.tensor.tensor_shape.dim]
                    vals = np.frombuffer(v.tensor.tensor_content, dtype=bool).reshape(shapes)
                    density_stats[v.tag][e.step] = vals

    with open(os.path.join(target_dir, 'density.pkl'), 'wb') as f:
        pkl.dump(dict(density_stats), f)

process_density()

