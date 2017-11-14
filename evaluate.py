import numpy as np
import os
import re

import eval.stats

if __name__ == '__main__':
    datadir = "/dfs/scratch0/rexy/graph_gen_data/"
    prefix = "GraphRNN_enzymes_50_26000_"
    real_graphs_filename = [f for f in os.listdir(datadir) if re.match('[' + prefix + '][real].*\.dat', f)]
    pred_graphs_filename = [f for f in os.listdir(datadir) if re.match('[' + prefix + '][pred].*\.dat', f)]
    print(real_graphs_filename)
    print(pred_graphs_filename)

