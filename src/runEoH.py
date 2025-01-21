import torch
from eoh import eoh
from evaluate_adv import *
from eoh.utils.getParas import Paras
from display_adv import *

# Parameter initilization #
paras = Paras()

# Set parameters #
paras.set_paras(method="eoh",  # ['ael','eoh']
                problem="adv_examples",  # ['tsp_construct','bp_online']
                llm_api_endpoint="oa.api2d.net",  # set your LLM endpoint
                llm_api_key="fk228556-5C1hePre5crl1WACmzkNTEZnKG2EihEh",  # set your key
                llm_model="gpt-4o-2024-08-06",
                dataset="coco",
                model="mlliw",
                num_classes=80,
                target=torch.tensor([1] * 80),
                ec_pop_size=8,  # number of samples in each population
                ec_n_pop=20,  # number of populations
                exp_n_proc=8,  # multi-core parallel
                exp_debug_mode=False)

# # initilization
# evolution = eoh.EVOL(paras)
#
# # run
# evolution.run()

# 评估攻击
evaluate_many_adv(paras, 'gpt')

# # 对比原始图片和对抗样本
# display_adv_compare()