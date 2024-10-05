# Tools for analyzing for Ca data

## Installation
Create a conda environment
```python
mamba create -y -n ca ipykernel pyyaml matplotlib xarray tqdm pyside6 numba flammkuchen h5py defopt pyqtgraph qtpy seaborn napari -y -c conda-forge
conda activate ca
pip install git+https://github.com/janclemenslab/napari-tifffile-reader.git git+https://github.com/janclemenslab/ca_utils
```

## Usage
See `demo/demo.ipynb`.