import torch
import os
from util import get_data
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pickle
from moco import MoCo
import numpy as np
from multiprocessing import Process
from tqdm import tqdm
import argparse


NUM_PER_ROUND = 10000
BATCH_SIZE = 32


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="./models", help='Path of the model file(s) to be loaded.')
    parser.add_argument('-q', '--query_path', type=str,
                        help='The directory path of query protein file(s) (PDB format or mmCIF format). ')
    parser.add_argument('-tmp', '--tmp_dir', type=str, default="./tmp", help='The folder of temporary files')
    parser.add_argument('-o', '--output_path', type=str, default="./output", help="Path of output directory.")
    parser.add_argument('-n', '--n_jobs', type=int, default=16, help="Number of CPU processes to use.")
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")

    return parser.parse_args()


class QuerySet(Dataset):
    def __init__(self, result_dict):
        super(QuerySet, self).__init__()
        self.esm_feature = result_dict['esm_feature_dict']
        self.raw_feature = result_dict['raw_data_dict']
        self.query_list = list(self.esm_feature.keys())

    def len(self) -> int:
        return len(self.query_list)

    def get(self, idx: int) -> Data:
        query = self.query_list[idx]
        data = Data(x=torch.tensor(self.raw_feature[query]['x']),
                    edge_index=torch.tensor(self.raw_feature[query]['edge_index']),
                    esm_feature=torch.tensor(self.esm_feature[query]).view(1, -1),
                    name=query)
        return data


def generate_descriptors(result_dict, output_dir, device, round=-1):
    descriptors_dict = {}
    net = MoCo(in_channels=32, graph_out_channels=512, esm_out_channels=512, dim=512, device=device)
    net.to(device)
    model_path = "./models"
    model_list = os.listdir(model_path)

    dataset = QuerySet(result_dict)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    for n in range(len(model_list)):
        weights = torch.load(os.path.join(model_path, model_list[n]), map_location="cpu")
        net.load_state_dict(weights)
        with torch.no_grad():
            net.eval()
            for _, data_batch in enumerate(dataloader):
                data_batch.to(device)
                descriptors = np.array(net(data_batch, test=True).cpu())
                for k in range(len(data_batch.name)):
                    if n == 0:
                        descriptors_dict[data_batch.name[k]] = descriptors[k, :]
                    elif n == len(model_list) - 1:
                        descriptors_dict[data_batch.name[k]] = (descriptors_dict[data_batch.name[k]]
                                                                + descriptors[k, :]) \
                                                               / len(model_list)
                    else:
                        descriptors_dict[data_batch.name[k]] = descriptors_dict[data_batch.name[k]] \
                                                               + descriptors[k, :]
    if round == -1:
        output_file = os.path.join(output_dir, f"descriptors.pkl")
    else:
        output_file = os.path.join(output_dir, f"descriptors_{round+1}.pkl")
    if os.path.exists(output_file):
        print(f"Warning: {output_file} already exits and will be overwriten.")
    with open(output_file, "wb") as pkl:
        pickle.dump(descriptors_dict, pkl)


def main():
    args = parse_argument()
    torch.multiprocessing.set_start_method("spawn")
    query_dir = args.query_path
    tmp_dir = args.tmp_dir
    output_dir = args.output_path
    device = torch.device('cpu')
    if torch.cuda.is_available() and not args.nogpu:
        device = torch.device('cuda')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    query_list = os.listdir(query_dir)
    num = len(query_list)
    if num > NUM_PER_ROUND:
        rounds = num // NUM_PER_ROUND + 1
        if num % NUM_PER_ROUND == 0:
            rounds = num / NUM_PER_ROUND
        print(f"Totally {num} structures in the query directory. "
              f"They will be divided into {rounds} rounds, "
              f"{NUM_PER_ROUND} per round.")
        for i in tqdm(range(rounds)):
            sub_query_list = query_list[i*NUM_PER_ROUND: (i+1)*NUM_PER_ROUND]
            result_dict = get_data(query_dir, sub_query_list, tmp_dir, n_process=args.n_jobs, n_gpu_process=1, device=device)
            sub_process = Process(target=generate_descriptors, args=(result_dict, output_dir, device, i,))
            sub_process.start()
            sub_process.join()

    else:
        print(f"Totally {num} structures in the query directory. ")
        print("Preparing data......")
        result_dict = get_data(query_dir, query_list, tmp_dir, n_process=args.n_jobs, n_gpu_process=1, device=device)
        print("Generating descriptors......")
        subprocess = Process(target=generate_descriptors, args=(result_dict, output_dir, device))
        subprocess.start()
        subprocess.join()
        print("Finished!")
if __name__ == "__main__":
    main()