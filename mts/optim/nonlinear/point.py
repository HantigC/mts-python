from mts.optim.gaussnewton import GaussNewton
from mts.optim.jr.point import PointJR


gn_point = GaussNewton(PointJR()).compute
