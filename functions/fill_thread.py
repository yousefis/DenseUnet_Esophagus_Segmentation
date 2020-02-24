import  threading, time

import functions.settings as settings
class fill_thread(threading.Thread):
    def __init__ (self, CTs, GTVs,train_Torso,Penalize,_image_class,
                  sample_no,total_sample_no,patch_window,GTV_patchs_size,
                  img_width,img_height,mutex,tumor_percent,other_percent,is_training,patch_extractor,fold):
        """
            Thread for moving images to RAM.

            This thread moves the images to RAM for train and validation process simultaneously fot making this process co-occurrence.

            Parameters
            ----------
            arg1 : int
                Description of arg1
            arg2 : str
                Description of arg2

            Returns
            -------
            nothing


        """
        threading.Thread.__init__(self)

        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())
        self.CTs=CTs
        self.GTVs=GTVs
        self.Penalize=Penalize
        self.Torsos=train_Torso
        self._image_class=_image_class
        self.sample_no=sample_no
        self.patch_window=patch_window
        self.GTV_patchs_size=GTV_patchs_size
        self.img_width=img_width
        self.img_height=img_height
        self.mutex=mutex
        self.tumor_percent=tumor_percent
        self.other_percent=other_percent
        self.is_training=is_training
        self.total_sample_no=total_sample_no
        self.patch_extractor=patch_extractor
        self.fold=fold

        self.Kill=False


    def run(self):
        while True:
            with self.pause_cond:
                while self.paused:
                    self.pause_cond.wait()
                # print('try aquired 1')
                # self.mutex.acquire()
                # print('aquired 1')
                try:
                    # if self.Kill:
                    #     settings.read_patche_mutex_vl.release()
                    #     break
                    # thread should do the thing if not paused
                    if self.is_training==0: #validation!
                        delta=10
                        if len(settings.bunch_GTV_patches_vl) > settings.validation_totalimg_patch:
                            break
                        # if len(settings.bunch_GTV_patches_vl)>self.total_sample_no-delta:
                        #     break
                        if settings.vl_isread == False:
                            continue
                        if self.fold<0:
                            self._image_class.read_bunch_of_images_vl_from2nd_dataset(settings.validation_totalimg_patch)
                        elif self.fold>100:
                            self._image_class.read_bunch_of_images_vl_from2nd_dataset(
                                settings.validation_totalimg_patch)
                        else:
                            self._image_class.read_bunch_of_images_vl_from_both_datasets(
                                settings.validation_totalimg_patch)
                        self.patch_extractor.resume()

                    else:#train

                        # while len(settings.bunch_GTV_patches)>300:
                        #     print('sleep bunch_GTV_patches:%d', len(settings.bunch_GTV_patches))
                        #     time.sleep(3)
                        if settings.tr_isread == False:
                            continue

                        if self.fold<0: #just read from 2nd dataset
                            self._image_class.read_bunch_of_images_from2nd_dataset()
                        elif self.fold>100: #just read from 1st dataset
                            self._image_class.read_bunch_of_images_from2nd_dataset()
                        else: #read from both dataset
                            self._image_class.read_bunch_of_images_from_both_dataset() # for training
                        self.patch_extractor.resume()
                    # self._image_class.read_patche_online_from_image_bunch(self.sample_no, self.patch_window,
                    #                                                       self.GTV_patchs_size,self.tumor_percent,self.other_percent)
                finally:
                    a=1
                    # self.mutex.release()

                    time.sleep(1)
                    # print('realsed 1')

                # print('do the thing')
            # self.pause()




    def pop_from_queue(self):
        return self.queue.get()
    def kill_thread(self):
        self.Kill=True

    def pause(self):
        # print('pause fill ')

        # If in sleep, we acquire immediately, otherwise we wait for thread
        # to release condition. In race, worker will still see self.paused
        # and begin waiting until it's set back to False
        self.pause_cond.acquire()
        # should just resume the thread
        self.paused = True

    def resume(self):
        # print('resume fill ')

        # Notify so thread will wake after lock released
        if self.paused :
            self.pause_cond.notify()
            # Now release the lock
            self.pause_cond.release()
            self.paused = False

