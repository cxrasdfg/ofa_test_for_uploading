# coding=utf-8
import numpy as np

class MobileNetV3Gene():
    
    def __init__(self, supernet) -> None:
        super().__init__()

        self.net = supernet
        

        ks_candidates = self.net.ks_list if self.net.__dict__.get('_ks_include_list', None) is None \
            else self.net.__dict__['_ks_include_list']
        expand_candidates = self.net.expand_ratio_list if self.net.__dict__.get('_expand_include_list', None) is None \
            else self.net.__dict__['_expand_include_list']
        depth_candidates = self.net.depth_list if self.net.__dict__.get('_depth_include_list', None) is None else \
            self.net.__dict__['_depth_include_list']

        self.lb = []
        self.ub =[]
        if not isinstance(ks_candidates[0], list):
            self.ks_candidates = [ks_candidates for _ in range(len(self.net.blocks) - 1)]
            self.lb += [0 for _ in range(len(self.net.blocks) - 1)]
            self.ub += [len(ks_candidates)-1 for _ in range(len(self.net.blocks) - 1)]

        
        if not isinstance(expand_candidates[0], list):
            self.expand_candidates = [expand_candidates for _ in range(len(self.net.blocks) - 1)]  
            self.lb += [0 for _ in range(len(self.net.blocks) - 1)]
            self.ub += [len(expand_candidates)-1 for _ in range(len(self.net.blocks) - 1)]

        if not isinstance(depth_candidates[0], list):
            self.depth_candidates = [depth_candidates for _ in range(len(self.net.block_group_info))]
            self.lb += [0 for _ in range(len(self.net.block_group_info))]
            self.ub += [len(depth_candidates)-1 for _ in range(len(self.net.block_group_info))]
        
        self.lb = np.array(self.lb)
        self.ub = np.array(self.ub)

    def get_n_var(self):
        return len(self.ub)
    

    def decode_arch(self, chromosome, instant_act_subnet=False):
        # from chromosome to the settings in mobile networks.
        ks_setting = []
        
        index_offset = 0
        for i, k_set in enumerate(self.ks_candidates):
            k = k_set[int(chromosome[i+index_offset])]
            ks_setting.append(k)
            
        index_offset += len(self.ks_candidates)

        # sample expand ratio
        expand_setting = []
     
        for i, e_set in enumerate(self.expand_candidates):
            e = e_set[int(chromosome[i+index_offset])]
            expand_setting.append(e)
        index_offset += len(self.ks_candidates)

        # sample depth
        depth_setting = []
        
        for i, d_set in enumerate(self.depth_candidates):
            d = d_set[int(chromosome[i+index_offset])]
            depth_setting.append(d)
        
        index_offset += len(self.depth_candidates)
        
        if instant_act_subnet:
            self.net.set_active_subnet(ks_setting, expand_setting, depth_setting)

        return {
            'ks': ks_setting,
            'e': expand_setting,
            'd': depth_setting,
        }

    def encode_arch(self, ks_setting, expand_setting, depth_setting):
        chromosome = []
        
        for k_set, k in zip(self.ks_candidates, ks_setting):
            idx = k_set.index(k)
            chromosome.append(idx)

        for e_set, e in zip(self.expand_candidates, expand_setting):
            idx = e_set.index(e)
            chromosome.append(idx)

        for d_set, d in zip(self.depth_candidates, depth_setting):
            idx = d_set.index(d)
            chromosome.append(idx)
        
        return np.array(chromosome)
    
