from alg.fedavg import fedavg
from alg.fedprox import fedprox
from alg.fedbn import fedbn
from alg.base import base
from alg.fedap import fedap
from alg.metafed import metafed
from alg.clusterfed import clusterfed
#from alg.fedmd import fedmd
from alg.fedbn_nt import fedbn_nt
from alg.fedavg_nt import fedavg_nt
from alg.fedprox_nt import fedprox_nt
from alg.fedmlb import fedmlb
from alg.fedamp import fedamp
from alg.fedmlb_nt import fedmlb_nt
from alg.fedbuab import fedbuab
from alg.fedicfa import fedicfa
#from alg.fedourcfl import fedourcfl
#from alg.fedavg_proto import fedavg_proto
from alg.fedcfl import fedcfl
from alg.fedpacfl import fedpacfl
#from alg.fedpre import fedpre
#from alg.fedHPer import fedHPer
#from alg.fedlama import fedlama
from alg.fedmysoft import fedmysoft
from alg.fedmysoft_nt import fedmysoft_nt
from alg.myfed import myfed

ALGORITHMS = [
    'myfed',
    'fedavg',
    'fedprox',
    'fedbn',
    'base',
    'fedap',
    'metafed',
    'clusterfed',
    'fedmd',
    'fedcd',
    'fedavg_nt',
    'fedprox_nt',
    'fedbn_nt',
    'fedmlb',
    'fedmlb_nt',
    'fedamp',
    'fedbuab',
    'fedicfa',
    'fedourcfl',
    'fedavg_proto',
    'fedcfl',
    'fedpacfl',
    'fedpre',
    'fedHPer',
    'fedlama',
    'fedmysoft'
    'fedmysoft_nt'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():

        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
