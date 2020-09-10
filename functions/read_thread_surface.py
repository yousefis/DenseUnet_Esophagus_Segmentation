import  threading, time
import numpy as np
import functions.settings as settings

class read_thread(threading.Thread):
    '''
    This class reads the patches from a bunch of images in a parallel way
    '''

    def __init__ (self,_fill_thread,mutex,validation_sample_no=0,is_training=1):
        threading.Thread.__init__(self)
        self._fill_thread=_fill_thread
        self.mutex=mutex

        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())
        self.is_training=is_training
        self.validation_sample_no=validation_sample_no
    def run(self):
        while True:
            with self.pause_cond:
                while self.paused:
                    self.pause_cond.wait()
                # print('try aquired2')
                # self.mutex.acquire()
                # print('aquired2')
                try:
                    # while len(settings.bunch_GTV_patches) > 300:
                    #     print('sleep bunch_GTV_patches:%d', len(settings.bunch_GTV_patches))
                    #     time.sleep(3)
                    # thread should do the thing iflen( not paused
                    if self.is_training==1:
                        if len(settings.bunch_GTV_patches)==0 :
                            settings.train_queue.acquire()
                            settings.bunch_CT_patches=settings.bunch_CT_patches2.copy()
                            settings.bunch_GTV_patches=settings.bunch_GTV_patches2.copy()
                            settings.bunch_Penalize_patches=settings.bunch_Penalize_patches2.copy()
                            settings.bunch_Surface_patches=settings.bunch_Surface_patches2.copy()

                            settings.bunch_CT_patches2 = []
                            settings.bunch_GTV_patches2 = []
                            settings.bunch_Penalize_patches2 = []
                            settings.bunch_Surface_patches2 = []
                            settings.train_queue.release()
                            self._fill_thread.resume()
                            #time.sleep(2)
                        else:
                            if len(settings.bunch_CT_patches2) and len(settings.bunch_GTV_patches2)and len(settings.bunch_Penalize_patches2)and len(settings.bunch_Surface_patches2):
                                settings.train_queue.acquire()
                                settings.bunch_CT_patches = np.vstack((settings.bunch_CT_patches,settings.bunch_CT_patches2))
                                settings.bunch_GTV_patches = np.vstack((settings.bunch_GTV_patches,settings.bunch_GTV_patches2))
                                settings.bunch_Penalize_patches = np.vstack((settings.bunch_Penalize_patches,settings.bunch_Penalize_patches2))
                                settings.bunch_Surface_patches = np.vstack((settings.bunch_Surface_patches,settings.bunch_Surface_patches2))
                                settings.bunch_CT_patches2=[]
                                settings.bunch_GTV_patches2=[]
                                settings.bunch_Penalize_patches2 = []
                                settings.bunch_Surface_patches2 = []
                                settings.train_queue.release()
                                self._fill_thread.resume()

                    else:
                        if len(settings.bunch_CT_patches_vl) > settings.validation_totalimg_patch:
                            del settings.bunch_CT_patches_vl2
                            del settings.bunch_GTV_patches_vl2
                            del settings.bunch_Penalize_patches2
                            del settings.bunch_Surface_patches2
                            break
                        if ((len(settings.bunch_GTV_patches_vl) == 0) \
                                &(len(settings.bunch_GTV_patches_vl) == 0)\
                                &(len(settings.bunch_Penalize_patches) == 0)\
                                &(len(settings.bunch_Surface_patches) == 0)\
                                &(len(settings.bunch_CT_patches_vl2)>0)\
                                &(len(settings.bunch_Penalize_patches_vl2)>0)\
                                &(len(settings.bunch_Surface_patches_vl2)>0)\
                                &(len(settings.bunch_GTV_patches_vl2)>0)):
                            settings.bunch_CT_patches_vl = settings.bunch_CT_patches_vl2
                            settings.bunch_CT_patches_vl2 = []

                            settings.bunch_GTV_patches_vl = settings.bunch_GTV_patches_vl2
                            settings.bunch_GTV_patches_vl2 = []

                            settings.bunch_Penalize_patches_vl = settings.bunch_Penalize_patches_vl2
                            settings.bunch_Penalize_patches_vl2 = []

                            settings.bunch_Surface_patches_vl = settings.bunch_Surface_patches_vl2
                            settings.bunch_Surface_patches_vl2 = []
                            print('settings.bunch_CT_patches_vl lEN: %d' %(len(settings.bunch_CT_patches_vl )))
                        elif ((len(settings.bunch_GTV_patches_vl) > 0) \
                                &(len(settings.bunch_GTV_patches_vl) > 0)\
                                &(len(settings.bunch_Penalize_patches) > 0)\
                                &(len(settings.bunch_Surface_patches) > 0)\
                                &(len(settings.bunch_GTV_patches_vl2) > 0)\
                                &(len(settings.bunch_GTV_patches_vl2) > 0)\
                                &(len(settings.bunch_Penalize_patches_vl2) > 0)\
                                &(len(settings.bunch_Surface_patches_vl2) > 0)):
                            settings.bunch_CT_patches_vl = np.vstack((settings.bunch_CT_patches_vl,settings.bunch_CT_patches_vl2))
                            settings.bunch_CT_patches_vl2 = []

                            settings.bunch_GTV_patches_vl = np.vstack((settings.bunch_GTV_patches_vl,settings.bunch_GTV_patches_vl2))
                            settings.bunch_GTV_patches_vl2 = []

                            settings.bunch_Penalize_patches_vl = np.vstack((settings.bunch_Penalize_patches_vl, settings.bunch_Penalize_patches_vl2))
                            settings.bunch_Penalize_patches_vl2 = []

                            settings.bunch_Surface_patches_vl = np.vstack(
                                (settings.bunch_Surface_patches_vl, settings.bunch_Surface_patches_vl2))
                            settings.bunch_Surface_patches_vl2 = []
                            print('settings.bunch_CT_patches_vl lEN2: %d' % (len(settings.bunch_CT_patches_vl)))

                        if len(settings.bunch_GTV_patches_vl)<self.validation_sample_no:
                            if self._fill_thread.paused==True:
                                self._fill_thread.resume()
                                # time.sleep(2)
                        else:
                            # self.mutex.release()
                            self.finish_thread()
                finally:
                    a=1
                    # settings.mutex.release()
                    time.sleep(1)
                    # print('release2')


            # self.pause()


    def pop_from_queue(self):
        return self.queue.get()

    def pause(self):
        # print('pause read ')

        # If in sleep, we acquire immediately, otherwise we wait for thread
        # to release condition. In race, worker will still see self.paused
        # and begin waiting until it's set back to False

        self.pause_cond.acquire()
        # should just resume the thread
        self.paused = True

    def resume(self):
        # print('resume read ')
        if self.paused:
            # Notify so thread will wake after lock released
            self.pause_cond.notify()
            # Now release the lock
            self.pause_cond.release()
            self.paused = False

    def finish_thread(self):
        self.pause()

