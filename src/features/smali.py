import re
import pandas as pd
import numpy as np
from glob import glob
import os
from tqdm import tqdm
import sys
from itertools import combinations
from p_tqdm import p_map, p_umap
from scipy import sparse
import pickle

from src.utils import UniqueIdAssigner, replace_with_dict


class SmaliApp():
    LINE_PATTERN = re.compile('^(\.method.*)|^(\.end method)|^[ ]{4}(invoke-.*)', flags=re.M)
    INVOKE_PATTERN = re.compile(
        "(invoke-\w+)(?:\/range)? {.*}, "     # invoke
        + "(\[*[ZBSCFIJD]|\[*L[\w\/$-]+;)->"   # package
        + "([\w$]+|<init>).+"                 # method
    )

    def __init__(self, app_dir):
        self.app_dir = app_dir
        self.package = app_dir.split('\\')[-2]
        self.smali_fn_ls = sorted(glob(
            os.path.join(app_dir, 'smali*/**/*.smali'), recursive=True
        ))
        if len(self.smali_fn_ls) == 0:
            print('Skipping invalid app dir:', self.app_dir, file=sys.stdout)
            return
        self.info = self.extract_info()
        print("this is the self info")
        print(self.info)
        print("------------------")

    def _extract_line_file(self, fn):
        with open(fn) as f:
            data = SmaliApp.LINE_PATTERN.findall(f.read())
            if len(data) == 0: return None

        data = np.array(data)
        assert data.shape[1] == 3  # 'start', 'end', 'call'

        relpath = os.path.relpath(fn, start=self.app_dir)
        data = np.hstack((data, np.full(data.shape[0], relpath).reshape(-1, 1)))
        return data

    def _assign_code_block(df):
        df['code_block_id'] = (df.start.str.len() != 0).cumsum()
        print(df)
        return df

    def _assign_package_invoke_method(df):
        res = (
            df.call.str.extract(SmaliApp.INVOKE_PATTERN)
            .rename(columns={0: 'invocation', 1: 'library', 2: 'method_name'})
        )
        print(pd.concat([df, res], axis=1))
        return pd.concat([df, res], axis=1)

    def extract_info(self):
        agg = [self._extract_line_file(f) for f in self.smali_fn_ls]
        df = pd.DataFrame(
            np.vstack([i for i in agg if i is not None]),
            columns=['start', 'end', 'call', 'relpath']
        )

        df = SmaliApp._assign_code_block(df)
        df = SmaliApp._assign_package_invoke_method(df)

        # clean
        assert (df.start.str.len() > 0).sum() == (df.end.str.len() > 0).sum(), f'Number of start and end are not equal in {self.app_dir}'
        df = (
            df[df.call.str.len() > 0]
            .drop(columns=['start', 'end']).reset_index(drop=True)
        )

        # verify no nans
        extract_nans = df.isna().sum(axis=1)
        assert (extract_nans == 0).all(), f'nan in {extract_nans.values.nonzero()} for {self.app_dir}'
        # self.info.loc[self.info.isna().sum(axis=1) != 0, :]

        return df


class HINProcess():

    def __init__(self, csvs, out_dir, nproc=4, test_size=0.67):
        self.csvs = csvs
        self.out_dir = out_dir
        self.nproc = nproc
        self.test_size = test_size
        self.packages = [os.path.basename(csv)[:-4] for csv in csvs]
        print('Processing CSVs')
        self.infos = p_map(HINProcess.csv_proc, csvs, num_cpus=nproc) # read the csv from interim,infos contain all the dataframe in a list
        self.prep_ids()
        print('preping done')

    def prep_ids(self):
        print('Processing APIs', file=sys.stdout)
        self.API_uid = UniqueIdAssigner()
        for info in tqdm(self.infos):
            info['api_id'] = self.API_uid.add(*info.api)
            del info['api']

        self.APP_uid = UniqueIdAssigner()
        for package in self.packages:
            self.APP_uid.add(package)

    def csv_proc(csv):
        df = pd.read_csv(
            csv, dtype={'method_name': str}, keep_default_na=False
        )
        df['api'] = df.library + '->' + df.method_name

        # Save precious memory
        del df['relpath']
        del df['call']
        del df['method_name']
        return df

    def construct_graph_A(self):
        print('Constructing A matrix')
        
        A_mat = sparse.lil_matrix((len(self.APP_uid), len(self.API_uid)), dtype=np.int8)
        for i, info in tqdm(enumerate(self.infos), total=len(self.APP_uid)):
            A_mat[i, info.api_id] = 1

        # shape: (# of apps, # of unique APIs)
        # A_mat = sparse.csr_matrix(A_mat)
        return A_mat

    def construct_A_counts(self):
        print('Constructing counts matrix')

        counts_mat = sparse.lil_matrix((len(self.APP_uid), len(self.API_uid)), dtype=np.uint8)
        for i, info in tqdm(enumerate(self.infos), total=len(self.APP_uid)):
            v = info.api_id.value_counts()
            counts_mat[i, v.index] = v.values

        # shape: (# of apps, # of unique APIs)
        return counts_mat

    def _prep_graph_B( csv):
        outfile = csv[:-4] + '.B'
        if os.path.exists(outfile + '.npy'):
            return np.load(outfile + '.npy')
        func_pairs = lambda d: list(combinations(d.api_id.unique(), 2))
        edges = pd.DataFrame(
            info.groupby('code_block_id').apply(func_pairs).explode()
            .reset_index(drop=True).drop_duplicates().dropna()
            .values.tolist()
        ).values.T.astype('uint32')

        np.save(outfile, edges)
        return edges

    def prep_graph_BP(self):
        print('Preparing B', file=sys.stdout)
        Bs = p_map(HINProcess._prep_graph_B, self.csvs, num_cpus=28)
        return Bs

    def _build_coo(arr_ls, shape):

        print("this is arr_ls")
        #print(arr_ls)
        arr = np.hstack([a for a in arr_ls if a.shape[0] == 2])
        print("this is arr_ls after hstack")
        print(arr)
        print(shape)
        arr = np.hstack([arr, arr[::-1, :]])
        print("this is arr_ls after 2nd hstack")
        #print(arr)
        #print(shape)
        arr = np.unique(arr, axis=1)  # drop dupl pairs
        values = np.full(shape=arr.shape[1], fill_value=1, dtype='i1')
        sparse_arr = sparse.coo_matrix(
            (values, (arr[0], arr[1])), shape=shape
        )
        sparse_arr.setdiag(1)
        print("this is sparse_arr")
        #print(sparse_arr)
        return sparse_arr

    def construct_graph_BP(self, Bs,tr_apis,selected):
        shape = (len(tr_apis), len(tr_apis))
        # Replace original API IDs with condensed consecutive IDs
        idx_map = dict(zip(tr_apis, range(len(tr_apis))))
        count=0
        for i in range(len(Bs)):
            B = Bs[i]
            #P = Ps[i]
            count=count+1
            print(count)

            # Select pairs of connections where both APIs are in training set
            # Filtering
            mask_B = np.nonzero(np.isin(B, selected).sum(axis=0) == 2)[0]
            B = B[:, mask_B]
            B = replace_with_dict(B, idx_map)
            Bs[i] = B

            #mask_P = np.nonzero(np.isin(P, tr_apis).sum(axis=0) == 2)[0]
            #P = P[:, mask_P]
            #P = replace_with_dict(P, idx_map)
            #Ps[i] = P
        try:
            api_dictionary = open('api_dictionary','wb')
            pickle.dump(idx_map,api_dictionary)
            api_dictionary.close()
        except Exception as e:
            print(e)
            print(" api dictionary save failed")


       # print("this is the training apis")
       # print(tr_apis)
       # print(type(tr_apis))

       # try:
       #     apis_count = open('/home/osw_w20023366/YumRahul/api_const/train_other_feature/training_api','wb')
       #     pickle.dump(tr_apis,apis_count)
       #     api_dictionary.close()
       # except Exception as e:
       #     print(e)
       #     print("training api save failed")


        print('Constructing B', file=sys.stdout)
        B_mat = HINProcess._build_coo(Bs, shape).tocsc()
        #print('Constructing P', file=sys.stdout)
        #P_mat = HINProcess._build_coo(Ps, shape).tocsc()
        return B_mat

    def save_matrices_other(self):
        print('Saving matrices', file=sys.stdout)
        path = self.out_dir
        # sparse.save_npz(os.path.join(path, 'A_full'), self.A_mat_full)
        sparse.save_npz(os.path.join(path, 'A_tr'), self.A_mat_tr)
        sparse.save_npz(os.path.join(path, 'A_tst'), self.A_mat_tst)
        sparse.save_npz(os.path.join(path, 'counts_tr'), self.counts_mat_tr)
        sparse.save_npz(os.path.join(path, 'counts_tst'), self.counts_mat_tst)

        #sparse.save_npz(os.path.join(path, 'code_block_id_tr'), self.B_mat_tr)
        # sparse.save_npz(os.path.join(path, 'B_tst'), self.B_mat_tr)
        #sparse.save_npz(os.path.join(path, 'library_tr'), self.P_mat_tr)
        # sparse.save_npz(os.path.join(path, 'P_tst'), self.P_mat_tst)
    def save_matrices_B(self):
        path = self.out_dir
        sparse.save_npz(os.path.join(path, 'code_block_id_tr'), self.B_mat_tr)

    def save_info(self):
        print('Saving infos', file=sys.stdout)
        path = self.out_dir
        s_API = pd.Series(self.API_uid.value_by_id, name='api')
        s_APP = pd.Series(self.APP_uid.value_by_id, name='app')
        s_API.to_csv(os.path.join(path, 'APIs.csv'))
        s_APP.to_csv(os.path.join(path, 'APPs.csv'))

    def shuffle_split(self):
        len_apps = 9959
        np.random.seed(1)
        shfld_apps = np.random.choice(np.arange(len_apps), len_apps, replace=False)
        cutoff = int(len_apps * self.test_size)
        print(f'Number of apps: {len_apps}, cutoff: {cutoff}')
        assert cutoff != len_apps
        tr_apps = shfld_apps[:cutoff]
        tst_apps = shfld_apps[cutoff:]
        #tr_apis = np.nonzero(self.A_mat[tr_apps, :].sum(axis=0))[1]
        #assert np.sum(tr_apis) > 0
        #print(f'{len(self.API_uid)} APIs overall, {len(tr_apis)} in training')
        freq_apis = np.nonzero(self.A_mat[tr_apps, :].sum(axis=0) > 1)[1]
        assert np.sum(freq_apis) > 0
        print(f'{len(self.API_uid)} APIs overall, {len(freq_apis)} in training have appeared at least twice')
        return tr_apps, tst_apps,freq_apis

    def train_test_split(self,Bs,tr_apps):
       # tr_apps, tst_apps,tr_apis = self.shuffle_split()

       # self.A_mat = self.A_mat.tocsc()
       # self.A_mat = self.A_mat[:, tr_apis]  # condense training APIs
       # self.A_mat = sparse.csr_matrix(self.A_mat)
       # self.A_mat_tr = self.A_mat[tr_apps, :]
       # self.A_mat_tst = self.A_mat[tst_apps, :]

       # self.counts_mat = self.counts_mat.tocsc()
       # self.counts_mat = self.counts_mat[:, tr_apis]  # condense training APIs
       # self.counts_mat = sparse.csr_matrix(self.counts_mat)
       # self.counts_mat_tr = self.counts_mat[tr_apps, :]
       # self.counts_mat_tst = self.counts_mat[tst_apps, :]

        Bs_tr = [B for i, B in enumerate(Bs) if i in tr_apps]
        # Bs_tst = [B for i, B in enumerate(Bs) if i in tst_apps]
        #Ps_tr = [P for i, P in enumerate(Ps) if i in tr_apps]
        # Ps_tst = [P for i, P in enumerate(Ps) if i in tst_apps]

        return Bs_tr #add Bs_tr here
        # return tr_apps, tst_apps, tr_apis, Bs_tr, Bs_tst, Ps_tr, Ps_tst

    def run(self):
        #self.A_mat = self.construct_graph_A()
        #self.counts_mat = self.construct_A_counts()
        Bs= self.prep_graph_BP()
        #del self.infos[:]
       # try:
       #    del self.infos
       # except:
       #    print('already deleted')
        with open('/home/osw_w20023366/YumRahul/api_const/train_other_feature/tr_apps', 'rb') as file:
       # Call load method to deserialze
            tr_apps = pickle.load(file)
        with open('/home/osw_w20023366/YumRahul/api_const/train_other_feature/tr_api', 'rb') as file:
       # Call load method to deserialze
            tr_apis = pickle.load(file)
        with open('/home/osw_w20023366/YumRahul/api_const/reduce_api', 'rb') as file:
       # Call load method to deserialze
            reduce_api = pickle.load(file)
        print("length of reduce api")
        print(len(reduce_api))
        print(type(reduce_api))
        print(reduce_api)
        selected_api = np.where(reduce_api == True)[0]
        print(selected_api)
        print(len(selected_api))
        print("length of tr_apis")
        print(len(tr_apis))
        print(type(tr_apis))
        Bs_tr = self.train_test_split(Bs,tr_apps)
        #print("enumerating done")
       # self.tr_apps = tr_apps
       # self.tst_apps = tst_apps
       # try:
       #     training = open('/home/osw_w20023366/YumRahul/api_const/train_other_feature/tr_apps','wb')
       #     pickle.dump(tr_apps,training)
       #     api_dictionary.close()
       # except Exception as e:
       #     print(e)
       #     print("training app save failed")
       # try:
       #     testing = open('/home/osw_w20023366/YumRahul/api_const/train_other_feature/tst_apps','wb')
       #     pickle.dump(tst_apps,testing)
       #     api_dictionary.close()
       # except Exception as e:
       #     print(e)
       #     print("testing app save failed")
       # try:
       #     training = open('/home/osw_w20023366/YumRahul/api_const/train_other_feature/tr_api','wb')
       #     pickle.dump(tr_apis,training)
       #     api_dictionary.close()
       # except Exception as e:
       #     print(e)
       #     print("training api save failed")
       # s_API = pd.Series(self.API_uid.value_by_id, name='api')
       # s_API = s_API[tr_apis].reset_index(drop=True)
       # s_API.to_csv(os.path.join(self.out_dir, 'APIs.csv'))
       # #del s_API
        #del self.A_mat
        #del self.counts_mat
        #self.save_matrices_other()
        #with open('/home/osw_w20023366/YumRahul/api_const/training_api', 'rb') as file: 
        # Call load method to deserialze
            #tr_apis = pickle.load(file)
        self.B_mat_tr= self.construct_graph_BP(Bs_tr,tr_apis,selected_api)
        #self.B_mat_tst, self.P_mat_tst = self.construct_graph_BP(Bs_tst, Ps_tst, tr_apis)
        self.save_matrices_B()
        print("done.............")
