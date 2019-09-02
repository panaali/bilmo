from bilmo.scripts.config import Config
from fastai.metrics import accuracy, FBeta, MultiLabelFbeta
from fastai.torch_core import add_metrics, listify
from fastai.basic_train import LearnerCallback, Learner
import torch
from functools import partial
import logging
conf = Config.conf
log = logging.getLogger("cafa-logger")


class MultiLabelFBetaMax(LearnerCallback):
    "Computes the fbeta score for multilabel classification without Threashold"
    _order = -20

    def __init__(self,
                 learn,
                 beta=1,
                 eps=1e-15,
                 sigmoid=True,
                 average="micro"):
        super().__init__(learn)
        self.eps, self.sigmoid, self.average, self.beta2 = \
            eps, sigmoid, average, beta**2
        dvc = self.learn.data.device
        self.batch_max_tp = torch.tensor(0., device=dvc)
        self.batch_max_fp = torch.tensor(0., device=dvc)
        self.batch_max_fn = torch.tensor(0., device=dvc)
        self.batch_max_tn = torch.tensor(0., device=dvc)
        self.batch_max_pr = torch.tensor(0., device=dvc)
        self.batch_max_re = torch.tensor(0., device=dvc)
        self.batch_max_f1 = torch.tensor(0., device=dvc)
        self.batch_max_threashold = torch.tensor(0., device=dvc)

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
            self.learn.recorder.add_metric_names([
                'tpm', 'fpm', 'prm', 'rem', 'thm', 'f1m',
                'tp', 'fp', 'pr', 're', 'f1'
            ])

    def on_epoch_begin(self, **kwargs):
        dvc = self.learn.data.device
        self.total_max_tp = torch.tensor(0., device=dvc)
        self.total_max_fp = torch.tensor(0., device=dvc)
        self.total_max_fn = torch.tensor(0., device=dvc)
        self.total_max_tn = torch.tensor(0., device=dvc)
        self.total_max_pr = torch.tensor(0., device=dvc)
        self.total_max_re = torch.tensor(0., device=dvc)
        self.total_max_f1 = torch.tensor(0., device=dvc)
        self.total_max_threashold = torch.tensor(0., device=dvc)

        self.total_tp = torch.tensor(0., device=dvc)
        self.total_fp = torch.tensor(0., device=dvc)
        self.total_fn = torch.tensor(0., device=dvc)
        self.total_tn = torch.tensor(0., device=dvc)
        self.total_pr = torch.tensor(0., device=dvc)
        self.total_re = torch.tensor(0., device=dvc)
        self.total_f1 = torch.tensor(0., device=dvc)

    def on_batch_end(self, last_output, last_target, **kwargs):
        dvc = self.learn.data.device
        with torch.no_grad():
            pred, targ = (last_output.sigmoid() if self.sigmoid else
                            last_output), last_target

            thresholds = torch.arange(0, 1, 0.01, device=dvc)
            exp_pred = pred.expand(
                (thresholds.shape[0], pred.shape[0], pred.shape[1]))
            exp_targ = targ.expand(
                (thresholds.shape[0], targ.shape[0], pred.shape[1]))
            exp_thresholds = thresholds.expand(
                (pred.shape[0] * pred.shape[1],
                    thresholds.shape[0])).t().view(
                        (thresholds.shape[0], pred.shape[0], pred.shape[1]))
            new_pred = (exp_pred > exp_thresholds).float()
            exp_tp = exp_targ * new_pred
            exp_f1 = (2 * exp_tp.sum(2) / (new_pred + exp_targ + self.eps).sum(2))
            max_f1 = exp_f1.mean(1).max()
            max_threashold = thresholds[exp_f1.mean(1).argmax()]
            pred_max = (pred > max_threashold).float()

            tp_max = pred_max * targ
            self.batch_max_tp = tp_max.sum(1).mean()
            self.batch_max_fp = (pred_max - tp_max).sum(1).mean()
            self.batch_max_fn = (targ - tp_max).sum(1).mean()
            self.batch_max_tn = (1 + tp_max - targ - pred_max).sum(1).mean()
            self.batch_max_pr = (tp_max.sum(1) / (pred_max + self.eps).sum(1)).mean()
            self.batch_max_re = (tp_max.sum(1) / (targ + self.eps).sum(1)).mean()
            self.batch_max_f1 = (2 * tp_max.sum(1) /
                             (pred_max + targ + self.eps).sum(1)).mean()
            self.batch_max_threashold = max_threashold

            tp = pred * targ
            self.batch_tp = tp.sum(1).mean()
            self.batch_fp = (pred - tp).sum(1).mean()
            self.batch_fn = (targ - tp).sum(1).mean()
            self.batch_tn = (1 + tp - targ - pred).sum(1).mean()
            self.batch_pr = (tp.sum(1) / (pred + self.eps).sum(1)).mean()
            self.batch_re = (tp.sum(1) / (targ + self.eps).sum(1)).mean()
            self.batch_f1 = (2 * tp.sum(1) /
                             (pred + targ + self.eps).sum(1)).mean()

            self.total_max_tp += self.batch_max_tp
            self.total_max_fp += self.batch_max_fp
            self.total_max_fn += self.batch_max_fn
            self.total_max_tn += self.batch_max_tn
            self.total_max_pr += self.batch_max_pr
            self.total_max_re += self.batch_max_re
            self.total_max_f1 += self.batch_max_f1
            self.total_max_threashold += self.batch_max_threashold

            self.total_tp += self.batch_tp
            self.total_fp += self.batch_fp
            self.total_fn += self.batch_fn
            self.total_tn += self.batch_tn
            self.total_pr += self.batch_pr
            self.total_re += self.batch_re
            self.total_f1 += self.batch_f1

    def on_epoch_end(self, last_metrics, **kwargs): #just for validation phase
        tp_max = round(self.total_max_tp.item() / kwargs['num_batch'], 4)
        fp_max = round(self.total_max_fp.item() / kwargs['num_batch'], 4)
        fn_max = round(self.total_max_fn.item() / kwargs['num_batch'], 4)
        tn_max = round(self.total_max_tn.item() / kwargs['num_batch'], 4)
        pr_max = round(self.total_max_pr.item() / kwargs['num_batch'], 4)
        re_max = round(self.total_max_re.item() / kwargs['num_batch'], 4)
        f1_max = round(self.total_max_f1.item() / kwargs['num_batch'], 4)
        max_threashold = round(
            self.total_max_threashold.item() / kwargs['num_batch'], 2)

        tp = round(self.total_tp.item() / kwargs['num_batch'],4)
        fp = round(self.total_fp.item() / kwargs['num_batch'], 4)
        fn = round(self.total_fn.item() / kwargs['num_batch'], 4)
        tn = round(self.total_tn.item() / kwargs['num_batch'], 4)
        pr = round(self.total_pr.item() / kwargs['num_batch'], 4)
        re = round(self.total_re.item() / kwargs['num_batch'], 4)
        f1 = round(self.total_f1.item() / kwargs['num_batch'], 4)
        return add_metrics(last_metrics, [
            tp_max, fp_max, pr_max, re_max, max_threashold, f1_max,
            tp, fp, pr, re, f1])


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
        # For each class how many sample were correct?
        self.tp += m.sum(0).float()
        # For each class how many samples were considered positive?
        self.total_pred += pred.sum(0).float()
        # For each class how many samples were truely positive?
        self.total_targ += targ.sum(0).float()

    MultiLabelFbeta.on_batch_end = on_batch_end

    learn_cls.callbacks.append(
        MultiLabelFbeta(learn_cls,
                        average='micro',
                        beta=1,
                        thresh=conf['multiLabelF1_thresh'],
                        sigmoid=True))

    learn_cls.callbacks.append(
        MultiLabelFBetaMax(learn_cls,
                                     beta=1,
                                     sigmoid=True))
    learn_cls.callbacks.append(
        ProgressBarComment(
            learn_cls,
            comments=[
                # '|tp',('self.learn.multi_label_f_beta_max', '.batch_tp'),
                # '|fp',('self.learn.multi_label_f_beta_max', '.batch_fp'),
                # '|tn',('self.learn.multi_label_f_beta_max', '.batch_tn'),
                # '|fn',('self.learn.multi_label_f_beta_max', '.batch_fn'),
                '|prm',('self.learn.multi_label_f_beta_max', '.batch_max_pr'),
                '|rem',('self.learn.multi_label_f_beta_max', '.batch_max_re'),
                '|thm',('self.learn.multi_label_f_beta_max', '.batch_max_threashold'),
                '|f1m',('self.learn.multi_label_f_beta_max', '.batch_max_f1'),
                '|pr',('self.learn.multi_label_f_beta_max', '.batch_pr'),
                '|re',('self.learn.multi_label_f_beta_max', '.batch_re'),
                '|f1',('self.learn.multi_label_f_beta_max', '.batch_f1'),
            ])
    )  # change comments to metric name and evaluate the metrics name in the
