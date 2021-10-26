import sys
import json

from src.utils import prep_dir
from src.data.get_data import get_data
from src.features.build_features import build_features

DATA_PARAMS = 'config/data-params.json'



def load_params(fp):
    with open(fp) as fh:
        param = json.load(fh)

    return param
    
def main(targets):
    
    if 'test-project' in targets:
        print("entered")
        cfg = load_params(DATA_PARAMS)
        prep_dir(**cfg)
        # get_data(**cfg)
        build_features(**cfg)

if __name__ == '__main__':
    targets = sys.argv[1:]
    # print(targets)
    main(targets)
