from .smplpytorch import SMPL_Layer
from .smpl_generator import SMPLHGenerator
from .wrapper_pytorch import SMPL_MODEL_ROOT

def get_smpl(gender, hands, model_root=SMPL_MODEL_ROOT):
    "simple wrapper to get SMPL model"
    return SMPL_Layer(model_root=model_root,
               gender=gender, hands=hands)