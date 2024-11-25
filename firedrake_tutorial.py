from firedrake import *
from firedrake.pyplot import FunctionPlotter, tripcolor
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

mesh = UnitSquareMesh(40, 40, quadrilateral=True)