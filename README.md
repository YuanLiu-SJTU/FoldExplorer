# FoldExplorer
FoldExplorer: Fast and Accurate Protein Structure Search with Sequence-Enhanced Graph Embedding

## Preparation
FoldExplorer is implemented with Python3, so a Python3 (>=3.7) interpreter is required.
At first, download the source code of FoldExplorer from Github:
```bash
git clone https://github.com/YuanLiu-SJTU/FoldExplorer.git
```
Then, we recommend you to use a virtual environment, such as Anaconda, to install the dependencies of FoldExplorer:
``` bash
conda create -n FoldExplorer python=3.7
conda activate FoldExplorer
```
Pytorch is necessary and we recommend you to use GPU to accelerate the computation. If you use GPU, please install a gpu-based Pytorch and select the proper cudatoolkit version number that matches your platform. For example:
```bash
conda install pytorch==1.12.1 cudatoolkit=11.3 -c pytorch
```
If you do not want to use gpu, please isntall a cpu-only Pytorch.
```bash
conda install pytorch cpuonly -c pytorch
```
And then install other requirements simply run:
```bash
pip install -r requirements.txt
```
When you want to quit the virtual environment, just:
```bash
conda deactivate FoldExplorer
```

## Usage
### Generate descriptors for protein structures using FoldExplorer
If you only want to generate descriptors for protein structures, you can run the command as following
```
conda activate FoldExplorer
python extract.py -q examples
```
The program will automatically create a temporary folder './tmp/' to store temporary files and an output folder './output/' to store the results. More details can be seen by running:
```python
python extract.py -h
```



