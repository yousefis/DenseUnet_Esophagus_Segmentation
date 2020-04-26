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
                try: #threadread_patche_online_from_image_bunch_vl fills the queue of images
                    if self.is_training==0: #for validation
                        if len(settings.bunch_GTV_patches_vl) > settings.validation_totalimg_patch:
                            break
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

                    else:#for train
                        if settings.tr_isread == False:
                            continue

                        if self.fold<0: #just read from 2nd dataset; training on the 2nd dataset
                            self._image_class.read_bunch_of_images_from2nd_dataset()
                        elif self.fold>100: #just read from 1st dataset; training on the 1st dataset
                            self._image_class.read_bunch_of_images_from2nd_dataset()
                        else: #read from all datasets; training on all the datasets
                            self._image_class.read_bunch_of_images_from_all_datasets() # for training
                        self.patch_extractor.resume()
                finally: #thread sleeps for 1sec
                    time.sleep(1)





    def pop_from_queue(self):
        return self.queue.get()
    def kill_thread(self):
        self.Kill=True

    def pause(self):
        self.pause_cond.acquire()
        # should just resume the thread
        self.paused = True

    def resume(self):
        if self.paused :
            self.pause_cond.notify()
            # Now release the lock
            self.pause_cond.release()
            self.paused = False

