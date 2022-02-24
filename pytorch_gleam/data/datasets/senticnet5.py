# download from http://sentic.net/downloads/
# extract senticnet.zip
# save senticnet5.py as senticnet5_raw.py in pytorch_gleam.data.datasets
from warnings import warn

try:
    import pytorch_gleam.data.datasets.senticnet5_raw as senticnet5

    senticnet = senticnet5.senticnet
except ImportError:
    warn("SenticNet 5 not properly installed, skipping. See `pytorch_gleam.data.datasets.senticnet5`")
    senticnet = {}
