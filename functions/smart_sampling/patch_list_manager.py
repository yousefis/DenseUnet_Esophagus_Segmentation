import collections
import functions.settings as settings
class smart_patching:
    def __init__(self):
        self.node = collections.namedtuple('node','ssim' 'location')

    def append(self,ssim_val,location):
        n = self.node(ssim=ssim_val, location=location)
        settings.patch_list.append(n)

    def delete(self):
        return

