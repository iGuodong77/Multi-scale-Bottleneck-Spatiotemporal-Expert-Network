def accuracy_indicators(perction, truth, data_name):
    shape = truth.shape
    esp = 1e-6
    TN = TP = FN = FP = 0
    if data_name == "quyu1" or data_name == "quyu2" or data_name == "quyu3" or data_name == "quyu4" or data_name == "quyu5":
        for i in range(shape[0]):
            for j in range(shape[1]):
                if truth[i, j] == 1 and perction[i, j] == 1:
                    TN = TN + 1
                if truth[i, j] == 1 and perction[i, j] == 2:
                    FP = FP + 1
                if truth[i, j] == 2 and perction[i, j] == 1:
                    FN = FN + 1
                if truth[i, j] == 2 and perction[i, j] == 2:
                    TP = TP + 1
    elif data_name == "river" or data_name == "Hermiston" or data_name == "farm" or data_name == "third":
        for i in range(shape[0]):
            for j in range(shape[1]):
                if truth[i, j] == 0 and perction[i, j] == 0:
                    TN = TN + 1
                if truth[i, j] == 0 and perction[i, j] != 0:
                    FP = FP + 1
                if truth[i, j] == 1 and perction[i, j] == 0:
                    FN = FN + 1
                if truth[i, j] == 1 and perction[i, j] != 0:
                    TP = TP + 1
    total = TN + TP + FN + FP
    PRA = (TN + TP) / (total + esp)   # PRA即为OA分类总精确度

    Pr = TP / (TP + FP + esp)
    Re = TP / (TP + FN + esp)
    F1 = 2 * Pr * Re / (Pr + Re + esp)

    Po = PRA
    Pc = ((TN + FN) * (TN + FP) + (TP + FP) * (TP + FN)) / (total * total)
    Kappa = (Po - Pc) / (1 - Pc)
    metrix = {"TN":TN, "FN":FN, "TP":TP, "FP":FP}

    return PRA, Kappa, F1, Pr, Re, metrix


# import scipy.io as sio
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
#
# ground_true = sio.loadmat("/home/HDD/dataset/river/ground_truth.mat")["ground_truth"]
#
# # ground_true = ground_true + 1
# # ground_true[ground_true == 1] = 0
# # ground_true[ground_true == 3] = 1
# # sio.savemat("/home/HDD/dataset/santaBarbara/ground_truth_gai.mat", {'ground_truth': ground_true})
# #
# #
# # ground_true[ground_true == 2] = 255
# # ground_true[ground_true == 1] = 127
# plt.imshow(ground_true)
# plt.show()
# ground_true = ground_true*255
# cv2.imwrite('/home/HDD/dataset/river/ground_truth.png', ground_true)


