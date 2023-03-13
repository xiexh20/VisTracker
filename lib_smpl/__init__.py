from .smplpytorch import SMPL_Layer
from .smpl_generator import SMPLHGenerator


def get_smpl(gender, hands):
    "simple wrapper to get SMPL model"
    return SMPL_Layer(model_root='/BS/xxie2020/static00/mysmpl/smplh',
               gender=gender, hands=hands)