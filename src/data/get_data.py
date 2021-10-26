import os
from glob import glob
import random
import src.utils as utils
# import src.data.sampling as sampling

import src.data.decompile as apktool


class IngestionException(Exception):
    pass


class IngestionPipeline():
    def __init__(self, n, out_dir, nproc):
        self.n_remaining = n
        self.out_dir = out_dir
        self.nproc = nproc
        self.final_dirs = []
        self.run()

   
    #lis of apps,number of apps and /data/raw/class0
    @classmethod
    def from_apks(cls, fp_iter, n, out_dir):
        cls.n_remaining = n
        cls.out_dir = out_dir
        cls.final_dirs = []
        while cls.n_remaining > 0:  # modified run() with only decompile
            cls.n_failed = 0
            apk_fps = [next(fp_iter) for _ in range(cls.n_remaining)]
            smali_dirs = cls.step_decompile_apks(cls, apk_fps)
            cls.final_dirs += smali_dirs
            cls.n_remaining = 0
        return cls.final_dirs

    
    def step_decompile_apks(self, apk_fps):
        smali_dirs = apktool.mt_decompile_apks(apk_fps, self.out_dir, 2)
        self.n_failed += sum(1 for i in smali_dirs if i is None)
        smali_dirs = [i for i in smali_dirs if i is not None]
        return smali_dirs

   






#nproc is the number of process
def stage_apk(cls_i_cfg, out_dir, nproc):
    #random
    sampling_cfg = cls_i_cfg['sampling']

    #test-data/comics'
    external_dir = cls_i_cfg['external_dir']

    if cls_i_cfg['external_structure'] == 'flat':
        #['test-data/communication\\ABradio Czech Depeche Mode_v2_apkpure.com.apk', 'test-data/communication\\com.Facultad.Learning.apk']
        apk_fps = glob(os.path.join(external_dir, '*.apk'))
        if sampling_cfg['method'] == 'random':
            #n is the number of apps
            n = sampling_cfg['n']
            assert len(apk_fps) >= n
            fp_iter = iter(random.sample(apk_fps, len(apk_fps)))
            #list of apks,number of application, #data\raw\class0
            smali_dirs = IngestionPipeline.from_apks(fp_iter, n, out_dir)
        elif sampling_cfg['method'] == 'all':
            assert len(apk_fps) > 0
            print(f'Ingesting {len(apk_fps)} apks')
            smali_dirs = apktool.mt_decompile_apks(apk_fps, out_dir, 2)
            smali_dirs = [i for i in smali_dirs if i is not None]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return smali_dirs




#nproc is the number of process
def run_pipeline(data_cfg, nproc):
    final_dirs = {}
    
    #gettting class and config of each class
    for cls_i, cls_i_cfg in data_cfg.items():
        print(f"Ingesting {cls_i}")
        #data\raw\class0
        out_dir = utils.RAW_CLASSES_DIRS[cls_i]
        if cls_i_cfg['stage'] == "apk":
            smali_dirs = stage_apk(cls_i_cfg, out_dir, nproc)
        final_dirs[cls_i] = smali_dirs
    
    return final_dirs

def get_data(**config):
    """Main function of data ingestion. Runs according to config"""
    # Set number of process, default to 2,nproc - number of process
    nproc = config['nproc'] if 'nproc' in config.keys() else 2
    classes = run_pipeline(config['data_classes'], nproc)
    print({k: len(v) for k, v in classes.items()})
