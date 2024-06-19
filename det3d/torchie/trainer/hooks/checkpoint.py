from ..utils import master_only
from .hook import Hook
import os
import os.path as osp
from det3d import torchie

# TODO: Add HOOK

class CheckpointHook(Hook):
    def __init__(self, interval=1, save_optimizer=True, out_dir=None, **kwargs):
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.args = kwargs

    @master_only
    def after_train_epoch(self, trainer):
        if not self.every_n_epochs(trainer, self.interval):
            return

        if not self.out_dir:
            self.out_dir = trainer.work_dir

        trainer.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args
        )


class CheckpointValHook(Hook):
    def __init__(self, interval=1, save_optimizer=True, out_dir=None, **kwargs):
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.args = kwargs

    @master_only
    def after_val_epoch(self, trainer):
        if not self.every_n_epochs(trainer, self.interval):
            return

        if not self.out_dir:
            self.out_dir = trainer.work_dir

        if trainer.cur_miou_val > trainer.max_miou_val:
            torchie.symlink(trainer.filesaved, trainer.linkpath_best)
            files = os.listdir(self.out_dir)

            for file in files:
                if file.endswith('.pth') and file != 'latest.pth' and file != 'best.pth' and file != trainer.filesaved:
                    print(os.path.join(self.out_dir, file))
                    os.remove(os.path.join(self.out_dir, file))

            # trainer.save_checkpoint(
            #     self.out_dir, save_optimizer=self.save_optimizer, **self.args
            # )

            trainer.max_miou_val = trainer.cur_miou_val

        else:
            files = os.listdir(self.out_dir)

            for file in files:
                if file.endswith('.pth') and file != 'latest.pth' and file != 'best.pth' and file != trainer.filesaved and file != os.readlink(trainer.linkpath_best).split('/')[-1]:
                    print(os.path.join(self.out_dir, file))
                    os.remove(os.path.join(self.out_dir, file))
