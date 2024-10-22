import os
import sys
from glob import glob
import pandas as pd
# !pip install multiprocess
from p_tqdm import p_umap
from scipy import sparse
import src.utils as utils
from src.features.smali_new import SmaliApp, HINProcess
import pickle


def is_large_dir(app_dir, size_in_bytes=1e20):
    try:
        if utils.get_tree_size(app_dir) > size_in_bytes:
            return True
        return False
    except Exception as e:
        print("File not found")
        print(app_dir)
        print(e)
        return True


def process_app(app_dir, out_dir):
    print('inside process app')
    print(app_dir)
    if is_large_dir(app_dir):
        print(f'Error {app_dir} too big')
        return None
    try:
        package = app_dir.split('\\')[-2]
        print("this is package")
        print(package)
        out_path = os.path.join(out_dir,package + '.csv')
        if os.path.isfile(out_path):
            print("path exist")     
        else:
            print("file_not_exist extract csv")
            app = SmaliApp(app_dir)
            app.info.to_csv(out_path, index=None)
            package = app.package
            del app
    except Exception as e:
        print(f'Error extracting {app_dir}')
        print(e)
        return None
    return package, out_path


def extract_save(in_dir, out_dir, class_i, nproc):
    app_dirs = glob(os.path.join(in_dir, '*/'))
    print("this is app dirs inside extract save")
    print(app_dirs)
    assert len(app_dirs) > 0, in_dir
    print(f'Extracting features for {class_i}')
    # process_app(app_dirs, out_dir)
    meta = p_umap(process_app, app_dirs, [out_dir for i in range(len(app_dirs))], num_cpus=nproc, file=sys.stdout)
    meta = [i for i in meta if i is not None]
    packages = [t[0]for t in meta]
    csv_paths = [t[1]for t in meta]
    return packages, csv_paths


def build_features(**config):
    print(config)
    """Main function of data ingestion. Runs according to config file"""
    # Set number of process, default to 2
    nproc = config['nproc'] if 'nproc' in config.keys() else 2
    # test_size = 0.67
    test_size = config['test_size'] if 'test_size' in config.keys() else 0.67


    csvs = []
    apps_meta = []
    for cls_i in utils.ITRM_CLASSES_DIRS.keys():
        #data\raw\class0
        raw_dir = utils.RAW_CLASSES_DIRS[cls_i]
        #data\interim\class0
        itrm_dir = utils.ITRM_CLASSES_DIRS[cls_i]
        # Look for processed csv files, skip extract step
        csv_paths = glob(f'{itrm_dir}/*.csv')
        
        if len(csv_paths) > 0:
            print('Found previously generated CSV files')
            packages = [os.path.basename(p)[:-4] for p in csv_paths]
        else:
            packages, csv_paths = extract_save(raw_dir, itrm_dir, cls_i, nproc)
        # Sort meta by package name for consistent index
        #dic of package csv path pair
        di = dict(zip(packages, csv_paths))
        # print(di)
        for package, csv_path in sorted(di.items()):
            apps_meta.append((cls_i, package, csv_path,))
            csvs.append(csv_path)
        

    print('Total number of csvs:', len(csvs))
    print("this are the csvs")
    #upto here extracting api of each app and storing it in iterm dir. list of csvs for all the extracted csvs
    print(csvs) 
    #list of interim csvs,data\\processed,2,0.67
    hin = HINProcess(csvs, utils.PROC_DIR,apps_meta,nproc=8, test_size=test_size)
    hin.run()

    # meta = pd.DataFrame(
    #     apps_meta,
    #     columns=['label', 'package', 'csv_path']
    # )
    # meta_train = meta.iloc[hin.tr_apps, :]
    # meta_train.index = [f'app_{i}' for i in range(len(meta_train))]
    # print('---')

    # meta_train.to_csv(os.path.join(utils.PROC_DIR, 'meta_tr.csv'))
    # meta_tst = meta.iloc[hin.tst_apps, :]
    # meta_tst.index = [f'app_{i + len(meta_train)}' for i in range(len(meta_tst))]
    # meta_tst.to_csv(os.path.join(utils.PROC_DIR, 'meta_tst.csv'))




