from mts.optim.gaussnewton import GaussNewton
from mts.optim.jr.rigid import PoseJR


gn_pose = GaussNewton(PoseJR()).compute
