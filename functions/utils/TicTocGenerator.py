import time
class TicTocGenerator(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print ('[%s]' % self.name,)
        print ('Elapsed: %s' % (time.time() - self.tstart))

# t = time.time()
# time.sleep(4)
# elapsed = time.time() - t
# print(elapsed)
# with TicTocGenerator('dd'):
#     time.sleep(4)
