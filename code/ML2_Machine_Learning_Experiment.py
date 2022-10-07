"""Code for performing a machine learning experiment using the experiment config file."""

import time
from configs.ML2_Configs import configs
from ML1_Machine_Learning import GraphML

for cf in configs:
    start = time.time()
    GraphML(cf, '../data/output/ML2/').run()
    print(f"{cf['net'].name} | {format((time.time() - start), '.2f')} sec")

print("Experiment was completed.")

