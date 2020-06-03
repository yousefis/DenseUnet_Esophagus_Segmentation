import collections
class smart_patching:
    def __init__(self):
        # self.node = collections.namedtuple('node','loss location')
        self.patch_list = []

    def append(self,ssim_val,location):
        n = self.patch_list.append((location, ssim_val))
        self.patch_list.patch_list.append(n)

    def refine(self):
        sorted(sorted(self.patch_list, key=lambda x: x[0]), key=lambda x: x[1], reverse=False)
        return

