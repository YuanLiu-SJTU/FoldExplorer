import multiprocessing
import os
from multiprocessing import Pool, Process
import subprocess
from Bio.PDB import PDBParser, MMCIFParser
import warnings
import math
import torch
import numpy as np
from esm import FastaBatchedDataset, pretrained


warnings.filterwarnings("ignore")
dist_threshold = 10
N_REF_POINTS = 31
longToShort = {'GLY': 'G',
               'ALA': 'A',
               'VAL': 'V',
               'LEU': 'L',
               'ILE': 'I',
               'PHE': 'F',
               'TRP': 'W',
               'TYR': 'Y',
               'ASP': 'D',
               'HIS': 'H',
               'ASN': 'N',
               'GLU': 'E',
               'MET': 'M',
               'MSE': 'M',
               'ARG': 'R',
               'SER': 'S',
               'THR': 'T',
               'CYS': 'C',
               'PRO': 'P',
               'SEC': 'U',
               'PYL': 'O',
               'LYS': 'K',
               'GLN': 'Q',
               'UNK': 'X'}


class GetFasta:
    def __init__(self, query_dir, query_list, tmp_dir, n_process):
        self.query_dir = query_dir
        self.output_dir = tmp_dir
        self.query_list = query_list
        self.n = n_process
        with open(os.path.join(self.output_dir, "proteins.fasta"), "w"):
            pass

    def get_single_fasta(self, pdb_path: str) -> str:
        try:
            struc = PDBParser().get_structure('protein', pdb_path)
            model = struc[0]
        except:
            struc = MMCIFParser().get_structure('protein', pdb_path)
            model = next(iter(struc), None)
        atom_ids = ('CA',)
        fasta = ""
        for chain in model:
            for res in chain:
                for atom in atom_ids:
                    if atom in res:
                        resname = res.get_resname()
                        if resname not in list(longToShort.keys()):
                            fasta += 'X'
                        else:
                            fasta += longToShort[res.get_resname()]
        return fasta

    def split_query_list(self):
        sub_query_lists = []
        m = len(self.query_list) // self.n
        for i in range(self.n-1):
            sub_query_lists.append(self.query_list[m*i:m*(i+1)])
        sub_query_lists.append(self.query_list[m*(self.n-1):])
        return sub_query_lists

    def write_fasta(self, fasta_list):
        with open(os.path.join(self.output_dir, "proteins.fasta"), "a") as f:
            f.writelines(fasta_list)

    def run_get_single_fasta(self, sub_query_list):
        fasta_list = []
        for query in sub_query_list:
            query_path = os.path.join(self.query_dir, query)
            fasta = self.get_single_fasta(query_path)
            fasta_list.append(">"+query+"\n"+fasta+"\n")
        return fasta_list

    def get_fasta_multiproc(self):
        pool_runfasta = Pool(self.n)
        sub_query_lists = self.split_query_list()
        for sub_query_list in sub_query_lists:
            pool_runfasta.apply_async(self.run_get_single_fasta, (sub_query_list,), callback=self.write_fasta)
        pool_runfasta.close()
        pool_runfasta.join()


class ESM:
    def __init__(self, fasta_file, tmp_dir, device, n_process=1):
        self.fasta = fasta_file
        self.tmp_dir = tmp_dir
        self.model_location = 'esm2_t33_650M_UR50D'
        self.n = n_process
        self.device = device

    def split_fasta_file(self):
        command = 'wc -l ' + self.fasta
        output = subprocess.check_output(command, shell=True)
        output_str = output.decode()
        lines = int(output_str.split()[0])
        lines_per_file = (lines // (self.n * 2) + 1) * 2
        command = f"split -l {lines_per_file} {self.fasta} -d -a 1 {self.tmp_dir}/fasta_"
        os.system(command)

    def run_esm2(self, fasta_file):
        model, alphabet = pretrained.load_model_and_alphabet(self.model_location)
        model.eval()
        model.to(self.device)

        dataset = FastaBatchedDataset.from_file(fasta_file)
        batches = dataset.get_batch_indices(4096, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches, pin_memory=True
        )

        result_dict = {}
        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                toks = toks.to(device=self.device, non_blocking=True)

                out = model(toks, repr_layers=[33], return_contacts=False)

                representations = {
                    layer: t.to(device="cpu") for layer, t in out["representations"].items()
                }
                for i, label in enumerate(labels):
                    truncate_len = min(1022, len(strs[i]))
                    result = {
                        layer: t[i, 1: truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                    result_dict[f"{label}"] = np.array(result[33])
        return result_dict

    def run_esm2_multiproc(self, result_dict):
        pool_runesm = Pool(self.n)
        results = []
        for i in range(self.n):
            fasta_file = os.path.join(self.tmp_dir, f"fasta_{i}")
            results.append(pool_runesm.apply_async(self.run_esm2, (fasta_file,)))
        pool_runesm.close()
        pool_runesm.join()
        esm_feature_dict = {}
        for result in results:
            esm_feature_dict.update(result.get())
        result_dict['esm_feature_dict'] = esm_feature_dict


class GAT:
    def __init__(self, query_dir, query_list, n_process):
        self.query_dir = query_dir
        self.query_list = query_list
        self.n = n_process

    def get_ca_coordinate(self, pdb_path: str) -> (np.ndarray, str):
        pdb = os.path.split(pdb_path)[-1]
        try:
            struc = PDBParser().get_structure('protein', pdb_path)
            model = struc[0]
        except:
            struc = MMCIFParser().get_structure('protein', pdb_path)
            model = next(iter(struc), None)
        atom_ids = ('CA',)
        fea = []
        fasta = ""
        for chain in model:
            for res in chain:
                for atom in atom_ids:
                    if atom in res:
                        fea.append(res[atom].coord)
                        resname = res.get_resname()
                        if resname not in list(longToShort.keys()):
                            fasta += 'X'
                        else:
                            fasta += longToShort[res.get_resname()]
        assert len(fea) == len(fasta), \
            "{} error: sequence length is not equal to node number(raw feature dimension).".format(pdb)
        return np.array(fea), fasta

    def extract_raw_features(self, xyz_list: np.ndarray, n_ref_points: int) -> (torch.tensor, torch.tensor):
        rxyz = self.get_relative_coordinate(xyz_list, n_ref_points)
        alphac_angle = self.cal_alphac_angle(xyz_list)
        fea = np.concatenate((rxyz, alphac_angle), axis=1)
        fea = torch.tensor(fea, dtype=torch.float32)
        p = np.array(xyz_list)
        vp = np.expand_dims(p, axis=1)
        dist_mat = torch.tensor(np.sqrt(np.sum(np.square(vp - p), axis=2)))
        adj_mat = (dist_mat < dist_threshold)
        return fea, adj_mat

    def get_relative_coordinate(self, xyz: np.ndarray, n_ref_points: int) -> np.ndarray:
        group_num = int(np.log2(n_ref_points + 1))
        assert 2 ** group_num - 1 == n_ref_points, \
            "The number of anchor points is {} and should be 2^k - 1, " \
            "where k is an integer, but k is {}.".format(n_ref_points, group_num)
        n_points = xyz.shape[0]
        ref_points = []
        for i in range(group_num):
            n_points_in_group = 2 ** i
            for j in range(n_points_in_group):
                beg, end = n_points * j // n_points_in_group, math.ceil(n_points * (j + 1) / n_points_in_group)
                ref_point = np.mean(xyz[beg:end, :], axis=0)
                ref_points.append(ref_point)
        coordinates = [np.linalg.norm(xyz - rp, axis=1).reshape(-1, 1) for rp in ref_points]
        return np.concatenate(coordinates, axis=1)

    def cal_alphac_angle(self, xyz: np.ndarray) -> np.ndarray:
        direction_vec = xyz[1:, :] - xyz[:-1, :]
        dv_1 = direction_vec[:-1, :]
        dv_2 = direction_vec[1:, :]
        dv_dot = np.sum(dv_1 * dv_2, axis=1)
        dv_norm = np.linalg.norm(dv_1, axis=1) * np.linalg.norm(dv_2, axis=1)
        pad_dv_norm = np.zeros((xyz.shape[0]))
        pad_dv_norm[1:-1] = dv_dot / dv_norm
        return pad_dv_norm.reshape((-1, 1))

    def split_query_list(self):
        sub_query_lists = []
        m = len(self.query_list) // self.n
        for i in range(self.n - 1):
            sub_query_lists.append(self.query_list[m * i:m * (i + 1)])
        sub_query_lists.append(self.query_list[m * (self.n - 1):])
        return sub_query_lists

    def run_single_proc(self, sub_query_list):
        sub_data_dict = {}
        for query in sub_query_list:
            pdb_path = os.path.join(self.query_dir, query)
            xyz, _ = self.get_ca_coordinate(pdb_path)
            feature, adj_mat = self.extract_raw_features(xyz, N_REF_POINTS)
            edge_index = torch.nonzero(adj_mat).permute(1, 0)
            data = {'x': np.array(feature),
                    'edge_index': np.array(edge_index),
                    'name': query}
            sub_data_dict[query] = data
        return sub_data_dict

    def run_multiproc(self, result_dict):
        sub_query_lists = self.split_query_list()
        pool_run = Pool(self.n)
        results = []
        for i in range(self.n):
            results.append(pool_run.apply_async(self.run_single_proc, (sub_query_lists[i],)))
        pool_run.close()
        pool_run.join()
        raw_data_dict = {}
        for result in results:
            sub_dict = result.get()
            raw_data_dict.update(sub_dict)
        result_dict['raw_data_dict'] = raw_data_dict


def get_data(query_dir, query_list, tmp_dir, n_process=4, n_gpu_process=1, device=torch.device('cpu')):
    getfasta = GetFasta(query_dir, query_list, tmp_dir, n_process=n_process)
    getfasta.get_fasta_multiproc()
    esm_runner = ESM(f"{tmp_dir}/proteins.fasta", tmp_dir, n_process=n_gpu_process, device=device)
    esm_runner.split_fasta_file()
    gat_runner = GAT(query_dir, query_list, n_process=n_gpu_process * 4)
    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    p1 = Process(target=esm_runner.run_esm2_multiproc, args=(result_dict,))
    p2 = Process(target=gat_runner.run_multiproc, args=(result_dict, ))

    p1.start()
    p2.start()
    p2.join()
    p1.join()

    return result_dict
