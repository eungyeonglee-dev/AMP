import numpy as np

model = ['bert','gpt2XL']
tp = [1,2,4]
gpu_type = ['A100','A10']
for m in model:
    for g in gpu_type:
        for t in tp:
            file_name = f"{m}_{g}_{t}.npy"
            data_1 = np.load(file_name)
            print(f"{file_name}, len: {len(data_1)}")
            print(data_1)