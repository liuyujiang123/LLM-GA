import json
import os

with open('/home/lyj/LLMs_Multi_Labels/results/history/gpt4_mlliw_voc2007_advance/population_generation_20.json', 'r') as f:
    data = json.load(f)

print(data['code'])
print(data['objective'])

# directory = "/home/lyj/LLMs_Multi_Labels/results/history/gpt4_mlliw_voc2007_advance"
# all_data = []
# for filename in os.listdir(directory):
#     if filename.endswith(".json"):
#         file_path = os.path.join(directory, filename)
#         with open(file_path, "r") as f:
#             data = json.load(f)
#             all_data.append(data['algorithm'])
# for algo in all_data:
#     print(algo)
