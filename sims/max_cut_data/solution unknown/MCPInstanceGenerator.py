'''
* Copyright (c) 2023 TECNALIA <esther.villar@tecnalia.com;eneko.osaba@tecnalia.com>
*
* This file is free software: you may copy, redistribute and/or modify it
* under the terms of the GNU General Public License as published by the
* Free Software Foundation, either version 3.
*
* This file is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

instance = 'MaxCut_20v3'
n_reduced_variables = 20
edges = 0

output = ""

for i in range(n_reduced_variables):
    for j in range(n_reduced_variables):
        if i != j and random.uniform(0,9) < 4:
            output = output + str(i+1) + " " + str(j+1) + " " + str(1)
            output = output + "\n"
            edges = edges+1

f = open(f'max_cut_data/{instance}.mc', "x")

f.write(str(n_reduced_variables) + " " + str(edges))
f.write("\n")
f.write(output)
f.close()