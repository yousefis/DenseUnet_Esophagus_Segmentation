import random
import numpy as np
class smart_patching:
    def __init__(self):
        # self.node = collections.namedtuple('node','loss location')
        self.patch_list = []
        self.max_elements = 6
        self.worst_patch_list=[]
        random.seed(90)
        self.children=[]


    def append(self,ssim_val,train_CT_image_patchs, train_GTV_label, train_Penalize_patch):
        self.patch_list.append((train_CT_image_patchs, train_GTV_label, train_Penalize_patch, ssim_val))


    def refine(self):
        if len(self.patch_list)<self.max_elements:
            return
        # if len(self.worst_patch_list):
        self.worst_patch_list=self.worst_patch_list+self.patch_list
        self.worst_patch_list =  (list(sorted(self.worst_patch_list, key=lambda x: x[-1],reverse=True)[0:self.max_elements]))
        # else:
        #     self.worst_patch_list = (sorted(self.patch_list, key=lambda x: x[-1])[0: self.max_elements])
        self.patch_list.clear()
        self.patch_list = []
    def intercourse(self):
        list_indv = list(range(len(self.worst_patch_list)))  # number of individuals
        random.shuffle(list_indv)  # individuals selection
        for i in range(int(len(list_indv)/2)):
            indv1 = self.worst_patch_list[list_indv[2 * i]]
            indv2 = self.worst_patch_list[list_indv[2 * i + 1]]
            randno = random.randint(1, len(indv1)-1)
            child1 = (np.concatenate((indv1[0][0:randno], indv2[0][randno:])),np.concatenate((indv1[1][0:randno], indv2[1][randno:])),np.concatenate((indv1[2][0:randno], indv2[2][randno:])))
            child2 = (np.concatenate((indv2[0][0:randno], indv1[0][randno:])),np.concatenate((indv2[1][0:randno], indv1[1][randno:])),np.concatenate((indv2[2][0:randno], indv1[2][randno:])))
            parents_indx = [list_indv[2 * i], list_indv[2 * i + 1]]  # indice of the parents
            self.children.append((parents_indx,child1, child2))
    def clear_lists(self):
        self.patch_list.clear()
        self.patch_list = []
        self.worst_patch_list.clear()
        self.worst_patch_list = []
        self.children.clear()
        self.children = []


if __name__=="__main__":
    sp = smart_patching()
    p=[4,2,5,1,6,9,4,7,4,3]
    for i in range(10):
        sp.append([[1,2,3],[5,6,6]],p[i])
    sp.refine()
