from bilmo.scripts.config import Config
from fastai.metrics import accuracy, FBeta, MultiLabelFbeta
from fastai.torch_core import add_metrics, listify
from fastai.basic_train import LearnerCallback, Learner
import torch
from functools import partial
import logging
conf = Config.conf
log = logging.getLogger("cafa-logger")


class ThresholdLessMultiLabelFBeta(LearnerCallback):
    "Computes the fbeta score for multilabel classification without Threashold"
    _order = -20

    def __init__(self,
                 learn,
                 beta=1,
                 eps=1e-15,
                 thresh=None,
                 sigmoid=True,
                 average="micro"):
        super().__init__(learn)
        self.eps, self.thresh, self.sigmoid, self.average, self.beta2 = \
            eps, thresh, sigmoid, average, beta**2
        dvc = self.learn.data.device
        self.batch_tp = torch.tensor(0., device=dvc)
        self.batch_fp = torch.tensor(0., device=dvc)
        self.batch_fn = torch.tensor(0., device=dvc)
        self.batch_tn = torch.tensor(0., device=dvc)
        self.batch_pr = torch.tensor(0., device=dvc)
        self.batch_re = torch.tensor(0., device=dvc)
        self.batch_f1 = torch.tensor(0., device=dvc)

    def on_train_begin(self, **kwargs):
        self.c = self.learn.data.c
        if self.average != "none":
            self.learn.recorder.add_metric_names(['tp', 'fp', 'fn', 'tn', 'f1'])

    def on_epoch_begin(self, **kwargs):
        dvc = self.learn.data.device
        self.total_tp = torch.tensor(0., device=dvc)
        self.total_fp = torch.tensor(0., device=dvc)
        self.total_fn = torch.tensor(0., device=dvc)
        self.total_tn = torch.tensor(0., device=dvc)
        self.total_f1 = torch.tensor(0., device=dvc)

    def on_batch_end(self, last_output, last_target, **kwargs):
        with torch.no_grad():
            pred, targ = (last_output.sigmoid() if self.sigmoid else
                            last_output), last_target
            if self.thresh is not None:
                pred, targ = (pred > self.thresh).byte(), targ.byte()

            tp = pred * targ
            self.batch_tp = tp.mean()
            self.batch_fp = (pred - tp).mean()
            self.batch_fn = (targ - tp).mean()
            self.batch_tn = (1 + tp - targ - pred).mean()
            self.batch_pr = (tp / (pred + self.eps)).mean()
            self.batch_re = (tp / (targ + self.eps)).mean()
            self.batch_f1 = (2 * tp/
                             (pred + targ + self.eps)).mean()
            self.total_tp += self.batch_tp
            self.total_fp += self.batch_fp
            self.total_fn += self.batch_fn
            self.total_tn += self.batch_tn
            self.total_f1 += self.batch_f1
            pass

    def on_epoch_end(self, last_metrics, **kwargs):
        tp = self.total_tp / kwargs['num_batch']
        fp = self.total_fp / kwargs['num_batch']
        fn = self.total_fn / kwargs['num_batch']
        tn = self.total_tn / kwargs['num_batch']
        f1 = self.total_f1 / kwargs['num_batch']
        return add_metrics(last_metrics, [tp, fp, fn, tn, f1])


class ProgressBarComment(LearnerCallback):
    _order = -9 # Run after Recorder to edit the progressbar comment

    def __init__(self,
                 learn: Learner,
                 comments: list):
        super().__init__(learn)
        self.comments = comments

    def on_backward_begin(self, **kwargs)->None:
        if self.learn.recorder.pbar is not None and hasattr(
                self.learn.recorder.pbar, 'child'):
            for comment in self.comments:
                if isinstance(comment, tuple):
                    comment = eval(comment[0] + comment[1])
                if isinstance(comment, torch.Tensor):
                    comment = f'{comment.item():.4f}'
                self.learn.recorder.pbar.child.comment += ' ' + comment  # change comments to metric name and evaluate the metrics name in the





def addf1MultiLabel(learn_cls):

    def on_batch_end(self, last_output, last_target, **kwargs):
        pred, targ = ((last_output.sigmoid() if self.sigmoid else last_output) > self.thresh).byte(), last_target.byte()
        # Which prediction was correct?
        m = pred*targ
        # For each class how many sample where correct?
        self.tp += m.sum(0).float()
        # For each class how many samples where considered positive?
        self.total_pred += pred.sum(0).float()
        # For each class how many samples where truely positive?
        self.total_targ += targ.sum(0).float()
        # log.debug('Average number of positive samples in this batch ' +
        #           str(round(pred.sum(0).float().mean().item(), 2)) +
        #           ' expected: ' +
        #           str(round(targ.sum(0).float().mean().item(), 2)) + ' ')

    MultiLabelFbeta.on_batch_end = on_batch_end

    learn_cls.callbacks.append(
        MultiLabelFbeta(learn_cls,
                        average='macro',
                        beta=1,
                        thresh=conf['multiLabelF1_thresh'],
                        sigmoid=True))

    learn_cls.callbacks.append(
        ThresholdLessMultiLabelFBeta(learn_cls,
                                     beta=1,
                                     thresh=None,
                                     sigmoid=True))
    learn_cls.callbacks.append(
        ProgressBarComment(
            learn_cls,
            comments=[
                '|tp' , ('self.learn.threshold_less_multi_label_f_beta', '.batch_tp'),
                '|fp' , ('self.learn.threshold_less_multi_label_f_beta', '.batch_fp'),
                '|tn' , ('self.learn.threshold_less_multi_label_f_beta', '.batch_tn'),
                '|fn' , ('self.learn.threshold_less_multi_label_f_beta', '.batch_fn'),
                '|pr' , ('self.learn.threshold_less_multi_label_f_beta', '.batch_pr'),
                '|re' , ('self.learn.threshold_less_multi_label_f_beta', '.batch_re'),
                '|f1' , ('self.learn.threshold_less_multi_label_f_beta', '.batch_f1'),
            ])
    )  # change comments to metric name and evaluate the metrics name in the
