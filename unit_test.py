# coding=utf-8

from ofa.ga.ga import NASSolver
from ofa.imagenet_classification.elastic_nn.networks.ofa_mbv3 import OFAMobileNetV3
from ofa.ga.ga import MobileNetV3Gene
from ofa.imagenet_classification.run_manager.distributed_run_manager_ga import DistributedRunManagerGA

distributed_run_manager = DistributedRunManager(
        args.path, net, run_config, compression, backward_steps=args.dynamic_batch_size, is_root=(hvd.rank() == 0)
    )

solver = NASSolver()

# horovodrun --verbose -np 2 -H localhost:2 python train_ofa_net_ga.py