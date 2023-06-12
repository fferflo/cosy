import os
if "PROJ_DATA" in os.environ:
    print(f"Warning from cosy: Environment variable PROJ_DATA already set to {os.environ['PROJ_DATA']}, overwriting")
os.environ["PROJ_DATA"] = os.path.join(os.path.dirname(__file__), "proj_data")

from .util import deduce_module, with_np

from .affine import *
from . import geo
from cosy.backend import proj
