import os
import sys

# seed = [42, 2022, 2024, 24, 66, 122, 446, 684, 3443, 2886, 1998, 1864, 4602, 2626, 6846, 8888, 88, 242, 388, 444]
seed = [2024, 66, 122, 42, 2886]
data_name = ['river', 'farm', 'bayArea']
margin = [0.1, 0.15, 0.2, 0.25, 0.3]
print(data_name)
for i in data_name:
    for m in margin:
        for j in seed:
            # for m in arg_1:
            print("------------------------------- Current data_name is {} ----------------------------------".format(i))
            print("------------------------------- Current margin is {} ----------------------------------".format(m))
            print("------------------------------- Current seed is {} ----------------------------------".format(j))
            # comment = 'python train_demo.py --arg_1 {} --data_name {} --seed {}'.format(m, j, i)
            comment = 'python train_demo.py --save_path "./results/abl/margin_{}/" --margin {} --data_name {} --seed {}'.format(m, m, i, j)
            os.system(comment)


# patch_size = [3, 5, 7, 9]
# # data_name = ['river', 'farm', "Hermiston", 'bayArea']
# data_name = ['bayArea', "Hermiston"]
# # margin = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
# for i in data_name:
#     for j in patch_size:
#         # for m in margin:
#         print("------------------------------- Current data_name is {} ----------------------------------".format(i))
#         print("------------------------------- Current patch_size is {} ----------------------------------".format(j))
#         # print("------------------------------- Current margin is {} ----------------------------------".format(m))
#         # comment = 'python train_demo.py --arg_1 {} --data_name {} --seed {}'.format(m, j, i)
#         comment = 'python train_demo.py --save_path "./results/" --data_name {} --patch_size {}'.format(i, j)
#         os.system(comment)