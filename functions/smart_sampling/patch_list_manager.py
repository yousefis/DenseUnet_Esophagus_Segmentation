import collections
class smart_patching:
    def __init__(self):
        # self.node = collections.namedtuple('node','loss location')
        self.patch_list = []
        self.max_elements = 20


    def append(self,ssim_val,location):
        self.patch_list.append((location, ssim_val))


    def refine(self):
        if len(self.patch_list)<self.max_elements:
            return
        self.patch_list = sorted(sorted(self.patch_list, key=lambda x: x[0]), key=lambda x: x[1], reverse=False)[0:self.max_elements]
    def mutation(self):
        if len(self.patch_list)<self.max_elements:
            return
        for i in range(self.patch_list):
            a=1


if __name__=="__main__":
    sp = smart_patching()
    p=[4,2,5,1,6,9,4,7,4,3]
    for i in range(10):
        sp.append([[1,2,3],[5,6,6]],p[i])
    sp.refine()
