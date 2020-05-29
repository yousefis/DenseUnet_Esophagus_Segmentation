import  threading, time
import functions.settings as settings
class _patch_extractor_thread(threading.Thread):
    def __init__ (self, _image_class,
                  sample_no,patch_window,GTV_patchs_size,
                  tumor_percent,other_percent,img_no,mutex,is_training,vl_sample_no=0):
        threading.Thread.__init__(self)

        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())
        self.mutex = mutex
        self.sample_no=sample_no
        self.patch_window=patch_window
        self.GTV_patchs_size=GTV_patchs_size
        if is_training:
            self._image_class=_image_class
        else:
            self._image_class_vl=_image_class
        self.tumor_percent=tumor_percent
        self.other_percent=other_percent
        self.img_no=img_no
        self.is_training=is_training
        self.validation_sample_no=vl_sample_no


    def run(self):
        while True:
            with self.pause_cond:
                while self.paused:
                    self.pause_cond.wait()
                # self.mutex.acquire()
                # print('try aquired 1')
                # print('aquired 1')
                # print('read_patche_online_from_image_bunch')
                try:
                    if self.is_training:
                        # while len(settings.bunch_GTV_patches)>300:
                        #     print('sleep bunch_GTV_patches:%d', len(settings.bunch_GTV_patches))
                        #     time.sleep(3)
                        self._image_class.read_patche_online_from_image_bunch_trackpatches(self.sample_no*10,
                                                                          self.patch_window,
                                                                          self.GTV_patchs_size,
                                                                          self.tumor_percent,
                                                                          self.other_percent,
                                                                          self.img_no
                                                                          )
                    else:

                        if len(settings.bunch_GTV_patches_vl) < settings.validation_totalimg_patch:
                            self._image_class_vl.read_patche_online_from_image_bunch_vl(self.sample_no,
                                                                              self.patch_window,
                                                                              self.GTV_patchs_size,
                                                                              self.tumor_percent,
                                                                              self.other_percent,
                                                                          14)
                        # else:
                        #     self.finish_thread()
                finally:
                    a=1
                    # self.mutex.release()
                    time.sleep(1)






    def pop_from_queue(self):
        return self.queue.get()

    def pause(self):
        # print('pause fill ')

        # If in sleep, we acquire immediately, otherwise we wait for thread
        # to release condition. In race, worker will still see self.paused
        # and begin waiting until it's set back to False
        self.pause_cond.acquire()
        # should just resume the thread
        self.paused = True

    def resume(self):
        # print('resume patch extr ')

        # Notify so thread will wake after lock released
        if self.paused :
            self.pause_cond.notify()
            # Now release the lock
            self.pause_cond.release()
            self.paused = False

    def finish_thread(self):
        self.pause()