import tensorflow as tf
import numpy as np
from scipy.ndimage import morphology
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


class _loss_func:
    def __init__(self):
        self.eps = 1e-6

    def weighted_cross_entropy_with_logits(self, targets, logits, name=None):
        pos_weight = .5  # self.get_class_ratio(targets,type_weight='Simple')
        with ops.name_scope(name, "logistic_loss", [logits, targets]) as name:
            logits = ops.convert_to_tensor(logits, name="logits")
            targets = ops.convert_to_tensor(targets, name="targets")
            try:
                targets.get_shape().merge_with(logits.get_shape())
            except ValueError:
                raise ValueError(
                    "logits and targets must have the same shape (%s vs %s)" %
                    (logits.get_shape(), targets.get_shape()))

            log_weight = 1 + (pos_weight - 1) * targets
            return math_ops.add(
                (1 - targets) * logits,
                log_weight * (math_ops.log1p(math_ops.exp(-math_ops.abs(logits))) +
                              nn_ops.relu(-logits)),
                name=name)

    def get_class_ratio(self, labels, type_weight):
        n_classes = labels.get_shape()[-1].value
        labels = tf.reshape(labels, [-1, n_classes])
        n_voxels = labels.get_shape()[0].value
        ref_vol = tf.reduce_sum(labels, 0)

        if type_weight == 'Square':
            weights = tf.reciprocal(tf.square(ref_vol))
        elif type_weight == 'Simple':
            weights = tf.reciprocal(ref_vol)
        elif type_weight == 'Uniform':
            weights = tf.ones_like(ref_vol)
        elif type_weight == 'Modified':
            weights = []
            for i in range(n_classes):
                weights.append((tf.reduce_sum(labels[:, i], 0)))
                weights = (tf.cast(weights, tf.float32) / tf.reduce_sum(labels))
        elif type_weight == 'Mean':
            weights = 1 / n_classes * tf.ones_like(ref_vol)

        new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
        weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                           tf.reduce_max(new_weights), weights)

        return weights

    def get_class_weights(self, labels, type_weight):
        n_classes = labels.get_shape()[-1].value
        labels = tf.reshape(labels, [-1, n_classes])
        n_voxels = labels.get_shape()[0].value
        ref_vol = tf.reduce_sum(labels, 0)

        if type_weight == 'Square':
            weights = tf.reciprocal(tf.square(ref_vol))
        elif type_weight == 'Simple':
            weights = tf.reciprocal(ref_vol)
        elif type_weight == 'Uniform':
            weights = tf.ones_like(ref_vol)
        elif type_weight == 'Modified':
            weights = []
            for i in range(n_classes):
                weights.append((tf.reduce_sum(labels[:, i], 0)))
                weights = (tf.cast(weights, tf.float32) / tf.reduce_sum(labels))
        elif type_weight == 'Mean':
            weights = 1 / n_classes * tf.ones_like(ref_vol)

        new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
        weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                           tf.reduce_max(new_weights), weights)

        return weights

    def tversky(self, logits, labels, weighting_flag=0, weighting_type='Square', alpha=0.5, beta=0.5):

        n_classes = logits.get_shape()[-1].value
        y_pred = tf.reshape(logits, [-1, n_classes])
        # y_pred = tf.nn.softmax(y_pred)
        y_true = tf.reshape(labels, [-1, n_classes])
        TP = tf.reduce_sum(y_pred * y_true, 0)
        TN = tf.reduce_sum((1.0 - y_true) * (1.0 - y_pred), 0)
        FP = tf.reduce_sum((1.0 - y_true) * y_pred, 0)
        FN = tf.reduce_sum((1.0 - y_pred) * y_true, 0)
        epsilon_denominator = self.eps
        if weighting_flag:
            class_weights = self.get_class_weights(labels, weighting_type)
            t_score = tf.reduce_sum(tf.multiply(class_weights, TP)) / tf.reduce_sum(
                tf.multiply(class_weights, (TP + alpha * FP + beta * FN + epsilon_denominator)))
        else:
            t_score = TP / (TP + alpha * FP + beta * FN + epsilon_denominator)

        return 1.0 - t_score
    def distance_based_penalty(self,logits, labels):
        non_zero_logit = tf.where(logits)

    def weighted_cross_entrophy_loss(self,logits,labels):
        pos_weight=2
        #weighted_cross_entropy_with_logits
        #,            pos_weight
        wce=tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels,
            logits=logits
        )
        return wce

    def new_loss(self, logits, labels, weighting_flag=0, weighting_type='Square',threshold=.5):
        n_classes = 2  # logits.get_shape()[-1].value

        # [TP,TN,FP,FN]=self.get_TP_TN_FP_FN( logits, labels)


        y_pred = tf.reshape(logits, [-1, n_classes])
        y_true = tf.reshape(labels, [-1, n_classes])

        y_pred = tf.nn.softmax(y_pred)

        subtract1 = tf.reduce_sum(y_pred * (1-y_true), 0)
        subtract2 = tf.reduce_sum((1-y_pred) * y_true, 0)
        intersect = tf.reduce_sum((subtract1), 0) + tf.reduce_sum(subtract2, 0)

        denominator = tf.reduce_sum((y_pred), 0) + tf.reduce_sum(y_true, 0)
        loss = ( intersect) / (denominator+ self.eps)

        return loss,y_pred,y_true

    def soft_dice(self, logits, labels, weighting_flag=0, weighting_type='Square',threshold=.5):
        n_classes = 2  # logits.get_shape()[-1].value

        # [TP,TN,FP,FN]=self.get_TP_TN_FP_FN( logits, labels)


        y_pred = tf.reshape(logits, [-1, n_classes])
        y_true = tf.reshape(labels, [-1, n_classes])

        y_pred = tf.nn.softmax(y_pred)

        intersect = tf.reduce_sum(y_pred * y_true, 0)

        denominator = tf.reduce_sum((y_pred), 0) + tf.reduce_sum(y_true, 0)
        if weighting_flag:
            class_weights = self.get_class_weights(labels, weighting_type)
            dice_scores = 2.0 * tf.reduce_sum(tf.multiply(class_weights, intersect)) / tf.reduce_sum(
                tf.multiply(class_weights, denominator + self.eps))
        else:
            dice_scores = (2.0 * intersect) / (denominator+ self.eps)

        return dice_scores,y_pred,y_true

    def surface_loss(self,logits, labels, surf_map):
        n_classes = 2
        y_pred = tf.reshape(logits, [-1, n_classes])
        y_true = tf.reshape(labels, [-1, n_classes])
        surf = tf.reshape(surf_map, [-1, 1])
        y_pred = tf.nn.softmax(y_pred)
        # loss = (y_true * (1 - y_pred) + y_pred * (1 - y_true))
        loss = tf.math.multiply((y_true * (1 - y_pred) + y_pred * (1 - y_true)) ,surf)
        return tf.reduce_sum(loss) /  (tf.reduce_sum(tf.cast(tf.where(labels[:,:,:,:,1]),tf.float32))+self.eps)

    def dice_plus_distance_penalize(self, logits, labels,penalize, weighting_flag=0, weighting_type='Square',threshold=.5):
        n_classes = 2
        y_pred = tf.reshape(logits, [-1, n_classes])
        y_true = tf.reshape(labels, [-1, n_classes])
        y_pred = tf.nn.softmax(y_pred)
        intersect = tf.reduce_sum(y_pred * y_true, 0)

        # penalized_loss=tf.reduce_sum(tf.math.divide(tf.math.multiply(tf.expand_dims(logits[:,:,:,:,1],-1),penalize),
        #                                                    tf.cast(tf.shape(logits)[0],tf.float32)))
        penalized_loss =  tf.math.multiply(tf.expand_dims(logits[:, :, :, :, 1], -1), penalize)
        penalized_loss = tf.reduce_sum((tf.math.multiply(penalized_loss,penalized_loss)))

        denominator = tf.reduce_sum((y_pred), 0) + tf.reduce_sum(y_true, 0)
        if weighting_flag:
            class_weights = self.get_class_weights(labels, weighting_type)
            dice_scores = 2.0 * tf.reduce_sum(tf.multiply(class_weights, intersect)) / tf.reduce_sum(
                tf.multiply(class_weights, denominator + self.eps))
        else:
            dice_scores = (2.0 * intersect) / (denominator+ self.eps)
        # loss= penalized_loss+dice_scores
        return  penalized_loss,dice_scores,y_pred,y_true
    def dice_plus_distance_penalize_focal_loss(self, logits, labels,penalize, weighting_flag=0, weighting_type='Square',threshold=.5):
        n_classes = 2
        y_pred = tf.reshape(logits, [-1, n_classes])
        y_true = tf.reshape(labels, [-1, n_classes])
        y_pred = tf.nn.softmax(y_pred)
        intersect = tf.reduce_sum(y_pred * y_true, 0)

        # penalized_loss=tf.reduce_sum(tf.math.divide(tf.math.multiply(tf.expand_dims(logits[:,:,:,:,1],-1),penalize),
        #                                                    tf.cast(tf.shape(logits)[0],tf.float32)))
        penalized_loss =  tf.math.multiply(tf.expand_dims(logits[:, :, :, :, 1], -1), penalize)
        penalized_loss = tf.reduce_sum((tf.math.multiply(penalized_loss,penalized_loss)))

        denominator = tf.reduce_sum((y_pred), 0) + tf.reduce_sum(y_true, 0)
        if weighting_flag:
            class_weights = self.get_class_weights(labels, weighting_type)
            dice_scores = 2.0 * tf.reduce_sum(tf.multiply(class_weights, intersect)) / tf.reduce_sum(
                tf.multiply(class_weights, denominator + self.eps))
        else:
            dice_scores = (2.0 * intersect) / (denominator+ self.eps)
        # loss= penalized_loss+dice_scores
        return  penalized_loss,tf.log(dice_scores+self.eps),y_pred,y_true
    def penalize_dice(self, logits, labels, penalize, weighting_flag=0, weighting_type='Square',threshold=.5):

        # y_pred_mask = penalize * logits
        # y_true_mask = penalize * labels
        # # [TP, TN, FP, FN] = self.get_TP_TN_FP_FN(logits=y_pred_mask, labels=y_true_mask)
        # intersect_0 = tf.reduce_sum(y_pred_mask * y_true_mask, 0)
        # denominator_0 = tf.reduce_sum((y_pred_mask), 0) + tf.reduce_sum(y_true_mask, 0)

        n_classes = 2  # logits.get_shape()[-1].value

        y_pred = tf.reshape(logits, [-1, n_classes])
        y_true = tf.reshape(labels, [-1, n_classes])

        y_pred = tf.nn.softmax(y_pred)

        intersect = tf.reduce_sum(y_pred * y_true, 0)
        denominator = tf.reduce_sum((y_pred), 0) + tf.reduce_sum(y_true, 0)

        FN = tf.reduce_sum((1.0 - y_pred) * y_true, 0)

        if weighting_flag:
            class_weights = self.get_class_weights(labels, weighting_type)
            dice_scores = 2.0 * tf.reduce_sum(tf.multiply(class_weights, intersect)) / tf.reduce_sum(
                tf.multiply(class_weights, denominator + self.eps))
        else:
            dice = (2.0 * intersect) / (denominator + self.eps)
            # edited_dice = (2.0 * intersect+2.0*intersect_0) / (denominator +denominator_0 + self.eps)
            edited_dice = (2.0 * intersect) / (denominator+FN + self.eps)

        return dice,edited_dice

    def f1_measure(self, logits, labels, alpha=1, beta=1):
        [TP, TN, FP, FN] = self.get_TP_TN_FP_FN(logits=logits, labels=labels)
        n_classes=2

        precision = TP / (TP + FP + self.eps)
        recall = TP / (TP + FN + self.eps)
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)  # f0:background, f1: tumor

        return f1

    def FPR(self, logits, labels):
        [TP, TN, FP, FN] = self.get_TP_TN_FP_FN(logits=logits, labels=labels)
        fpr = FP / (TN + FP + self.eps)
        return fpr

    def get_TP_TN_FP_FN(self, logits, labels):
        n_classes = logits.get_shape()[-1].value
        y_pred = tf.reshape(logits, [-1, n_classes])
        y_pred = tf.round(tf.nn.softmax(y_pred))
        y_true = tf.reshape(labels, [-1, n_classes])
        TP = tf.reduce_sum(y_pred * y_true, 0)
        TN = tf.reduce_sum((1.0 - y_true) * (1.0 - y_pred), 0)
        FP = tf.reduce_sum((1.0 - y_true) * y_pred, 0)
        FN = tf.reduce_sum((1.0 - y_pred) * y_true, 0)


        return TP, TN, FP, FN

    def FNR(self, logits, labels):
        [TP, TN, FP, FN] = self.get_TP_TN_FP_FN(logits=logits, labels=labels)
        fnr = FN / (TP + FN + self.eps)
        return fnr

    def dsc_fn(self, logits, labels, smooth):
        # convert the true labels to int64
        labels_binary = tf.to_int64(tf.argmax(labels, 4))
        # convert the predicted labels to int64
        logits_binary = tf.to_int64(tf.argmax(logits, 4))
        # calculate the intersection between true and predicted labels
        intersection = tf.reduce_sum(labels_binary * logits_binary)
        intersection = tf.cast(intersection, tf.float32)
        # calculate the union between true and predicted labels
        union = tf.reduce_sum(logits_binary) + tf.reduce_sum(labels_binary)
        union = tf.cast(union, tf.float32)
        # calculate the dice value
        dsc_value = (2 * intersection) / (union + smooth)
        dsc_value = tf.reduce_mean(tf.cast(dsc_value, tf.float32))

        return dsc_value

    def accuracy_fn(self, logits, labels):
        labels = tf.to_int64(labels)
        correct_prediction = tf.equal(tf.argmax(logits, 4), tf.argmax(labels, 4))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def PWC(self, logits, labels, ):
        [TP, TN, FP, FN] = self.get_TP_TN_FP_FN(logits=logits, labels=labels)

        pwc = (FN + FP) / (TP + FN + FP + TN + self.eps)
        return pwc

    def Precision(self, logits, labels):
        [TP, TN, FP, FN] = self.get_TP_TN_FP_FN(logits=logits, labels=labels)
        precision = TP / (TP + FP + self.eps)
        return precision

    def Recall(self, logits, labels):
        [TP, TN, FP, FN] = self.get_TP_TN_FP_FN(logits=logits, labels=labels)
        recall = TP / (TP + FN + self.eps)
        return recall

        # --------------------------------------------------------------------------------------------------------


    def surfd(self,input1, input2, sampling=1, connectivity=1):
        input_1 = np.atleast_1d(input1.astype(np.bool))
        input_2 = np.atleast_1d(input2.astype(np.bool))

        conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

        S = input_1 - morphology.binary_erosion(input_1, conn)
        Sprime = input_2 - morphology.binary_erosion(input_2, conn)

        dta = morphology.distance_transform_edt(~S, sampling)
        dtb = morphology.distance_transform_edt(~Sprime, sampling)

        sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

        return sds
