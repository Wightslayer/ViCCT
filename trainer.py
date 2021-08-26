import os
import time
import numpy as np
import torch
import csv
from torch.utils.tensorboard import SummaryWriter

from datasets.dataset_utils import img_equal_unsplit
import matplotlib.pyplot as plt
from matplotlib import cm as CM


class Trainer:
    def __init__(self, model, loading_data, cfg, cfg_data):
        """
         The Trainer is the class that facilitates the training of a model. After initialising, call 'train' to
        train the model. Based on the trainer of the C^3 Framework: https://github.com/gjy3035/C-3-Framework
        :param model: The model to be trained
        :param loading_data: a function with which the train/val/test dataloaders can be retrieved, as well as the
                             transform that transforms normalised images back to its original.
        :param cfg: The configurations for this run, specified in config.py
        :param cfg_data: The configurations specific to the dataset and dataloaders, specified in settings.py
        """

        self.model = model
        self.cfg = cfg  # General configuration of the run
        self.cfg_data = cfg_data  # Configuration specific to the dataloaders

        # Loading data makes the dataloaders for the training set, validation set, and testing set.
        # Also returns the restore transform with which the the un-normalised image can be obtained.
        self.train_loader, self.val_loader, self.test_loader, self.restore_transform = loading_data(self.model.crop_size)
        self.train_samples = len(self.train_loader.dataset)  # How many training samples we got
        self.val_samples = len(self.val_loader.dataset)  # How many validation samples we got

        # Saves one example predictions every this much evaluation samples.
        self.eval_save_example_every = self.val_samples // self.cfg.SAVE_NUM_EVAL_EXAMPLES

        self.criterion = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        # We take a step only at predefined epochs. Hence, step_size = 1.
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=cfg.LR_GAMMA)

        self.epoch = 0
        self.best_mae = 10 ** 10  # just something high
        self.best_epoch = -1  # We don't have a best epoch yet.

        self.writer = SummaryWriter(cfg.SAVE_DIR)  # For logging. We store vars like training/validation MAE and MSE.

        if cfg.RESUME:  # Should we resume training?
            self.load_state(cfg.RESUME_PATH)  # This loads and overwrites some important variables to continue training.
            print(f'Resuming from epoch {self.epoch}')
        else:
            # During evaluation we can save example predictions. This function save the corresponding images and GTs.
            self.save_eval_pics()
            self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], self.epoch)  # Log the current LR.

    def train(self):
        """ Trains the model.
        Also evaluates the model every 'EVAL_EVERY' epochs, and logs some informative metrics."""

        MAE, MSE, avg_val_loss = self.evaluate_model()
        print(f'Initial MAE: {MAE:.3f}, MSE: {MSE:.3f}, avg loss: {avg_val_loss:.3f}')

        while self.epoch < self.cfg.MAX_EPOCH:  # Train for MAX_EPOCH epochs
            self.epoch += 1

            epoch_start_time = time.time()  # Time how long an epoch takes
            # MACE = Mean Absolute Crop Error.
            # MSCE = Mean Squared Crop Error.
            losses, MACE, MSCE, last_out_den, last_gts = self.run_epoch()
            epoch_time = time.time() - epoch_start_time

            avg_train_loss = np.mean(losses)
            # The predicted count and GT counts for one random crop. Used for informative prints only!
            pred_cnt = last_out_den[0].detach().cpu().sum() / self.cfg_data.LABEL_FACTOR
            gt_cnt = last_gts[0].cpu().sum() / self.cfg_data.LABEL_FACTOR
            print(f'ep {self.epoch}: Average loss={avg_train_loss:.3f}, Patch MAE={MACE:.3f}, Patch MSE={MSCE:.3f}.'
                  f'  Example: pred={pred_cnt:.3f}, gt={gt_cnt:.3f}. Train time: {epoch_time:.3f}')

            # Logging training metrics in the summarywriter
            self.writer.add_scalar('AvgLoss/train', avg_train_loss, self.epoch)
            self.writer.add_scalar('MAE/train', MACE, self.epoch)
            self.writer.add_scalar('MSE/train', MSCE, self.epoch)

            # Evaluation
            if self.epoch % self.cfg.EVAL_EVERY == 0:  # Eval every 'EVAL_EVERY' epochs.
                eval_start_time = time.time()  # Time how long evaluation takes
                MAE, MSE, avg_val_loss = self.evaluate_model()
                eval_time = time.time() - eval_start_time

                if MAE < self.best_mae:  # New best Mean Absolute Error
                    self.best_mae = MAE
                    self.best_epoch = self.epoch
                    print_fancy_new_best_MAE()  # Super important. Gotta get that dopamine!
                    self.save_state(f'new_best_MAE_{MAE:.3f}')  # Save all states needed to continue training the model
                elif self.epoch % self.cfg.SAVE_EVERY == 0:  # save the state every 'SAVE_EVERY' regardless of the MAE
                    self.save_state(f'MAE_{MAE:.3f}')

                # Informative print
                print(f'MAE: {MAE:.3f}, MSE: {MSE:.3f}. best MAE: {self.best_mae:.3f} at ep({self.best_epoch}).'
                      f' eval time: {eval_time:.3f}')

                # Logging evaluation metrics in summarywriter
                self.writer.add_scalar('AvgLoss/eval', avg_val_loss, self.epoch)
                self.writer.add_scalar('MAE/eval', MAE, self.epoch)
                self.writer.add_scalar('MSE/eval', MSE, self.epoch)

            if self.epoch in self.cfg.LR_STEP_EPOCHS:  # Updates the learning rate
                self.scheduler.step()  # Make one update
                print(f'Learning rate adjusted to {self.scheduler.get_last_lr()[0]} at epoch {self.epoch}.')
                self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], self.epoch)

    def run_epoch(self):
        """ Run one pass over the train dataloader. """

        losses = []  # To compute the average loss over all predictions
        ACEs = []  # Absolute Crop Errors
        SCEs = []  # Squared Crop Errors

        out_den = None  # SILENCE WENCH!
        gt_stack = None  # Silences the 'might not be defined' warning below the for loop.

        self.model.train()  # Put model in training mode
        for idx, (img_stack, gt_stack) in enumerate(self.train_loader):
            img_stack = img_stack.cuda()  # A batch of training crops
            gt_stack = gt_stack.cuda()

            self.optim.zero_grad()
            out_den = self.model(img_stack)  # Make a prediction for the training crops
            loss = self.criterion(out_den, gt_stack)
            loss.backward()
            self.optim.step()

            losses.append(loss.cpu().item())  # The loss of this training batch
            errors = torch.sum(out_den - gt_stack, dim=(-2, -1)) / self.cfg_data.LABEL_FACTOR  # pred count - gt count
            ACEs.extend(torch.abs(errors).tolist())  # Absolute Crop Errors
            SCEs.extend(torch.square(errors).tolist())  # Squared Crop Errors

        MACE = np.mean(ACEs)  # Mean Absolute Crop Error
        MSCE = np.sqrt(np.mean(SCEs))  # Mean (Root) Squared Crop Error

        # Also return the last predicted densities and corresponding gts. This allows for informative prints.
        return losses, MACE, MSCE, out_den, gt_stack

    def evaluate_model(self):
        """ Evaluate the model on the evaluation dataloader. """

        plt.cla()  # Clear plot for new ones
        self.model.eval()
        with torch.no_grad():
            AEs = []  # Absolute Errors
            SEs = []  # Squared Errors
            losses = []  # To compute the average loss of all evaluation predictions

            abs_patch_errors = torch.zeros(self.model.crop_size, self.model.crop_size)  # For pixelwise error heatmap

            for idx, (img, img_stack, gt_stack) in enumerate(self.val_loader):
                img_stack = img_stack.squeeze(0).cuda()
                gt_stack = gt_stack.squeeze(0)  # Remove batch dim
                img = img.squeeze(0)  # Remove batch dim
                _, img_h, img_w = img.shape

                pred_den = self.model(img_stack)
                loss = self.criterion(pred_den, gt_stack.cuda())  # Just for logging. No gradients are computed here
                losses.append(loss.cpu().item())
                pred_den = pred_den.cpu()

                # The predictions are from image crops. Here, we reconstruct the density maps of the entire image.
                gt = img_equal_unsplit(gt_stack, self.cfg_data.OVERLAP, self.cfg_data.IGNORE_BUFFER, img_h, img_w, 1)
                den = img_equal_unsplit(pred_den, self.cfg_data.OVERLAP, self.cfg_data.IGNORE_BUFFER, img_h, img_w, 1)
                den = den.squeeze(0)  # Remove channel dim

                # The density maps are scaled by a LABEL FACTOR. To get the actual counts, reverse this scaling.
                pred_cnt = den.sum() / self.cfg_data.LABEL_FACTOR
                gt_cnt = gt.sum() / self.cfg_data.LABEL_FACTOR
                AEs.append(torch.abs(pred_cnt - gt_cnt).item())  # Store absolute error
                SEs.append(torch.square(pred_cnt - gt_cnt).item())  # Store squared error

                if idx % self.eval_save_example_every == 0:  # We only save a few examples
                    plt.imshow(den, cmap=CM.jet)  # Not actually displayed on the screen. Just to save the prediction
                    save_path = os.path.join(self.cfg.PICS_DIR, f'pred_{idx}_ep_{self.epoch}.jpg')
                    plt.title(f'Predicted count: {pred_cnt:.3f} (GT: {gt_cnt:.3f})')
                    plt.savefig(save_path)  # Save the prediction

                # Summed absolute error of each pixel of all crops. Gives insight in where most errors are made.
                abs_patch_errors += torch.sum(torch.abs(gt_stack.squeeze(1) - pred_den.squeeze(1)), dim=0)

            MAE = np.mean(AEs)  # Mean Absolute Error
            MSE = np.sqrt(np.mean(SEs))  # (root) Mean Squared Error
            avg_loss = np.mean(losses)

        plt.cla()  # Clear all plots (otherwise, things like titles will stay for new plots)
        plt.imshow(abs_patch_errors)  # The accumulated absolute error at each pixel in all crops
        save_path = os.path.join(self.cfg.PICS_DIR, f'errors_ep_{self.epoch}.jpg')
        plt.savefig(save_path)

        return MAE, MSE, avg_loss

    def save_eval_pics(self):
        """ During evaluation, some predictions are saved for visualisation. This function saves these images and their
        ground truth density maps.  This function is called at class initialisation."""

        plt.cla()  # Clear all plots
        for idx, (img, img_patches, gt_patches) in enumerate(self.val_loader):
            gt_patches = gt_patches.squeeze(0)  # Remove batch dim
            img = img.squeeze(0)  # Remove batch dim

            _, img_h, img_w = img.shape

            # The dataloader splits the ground truth into crops. Here, we restore the original GT density map.
            gt = img_equal_unsplit(gt_patches, self.cfg_data.OVERLAP, self.cfg_data.IGNORE_BUFFER, img_h, img_w, 1)
            gt = gt.squeeze()  # Remove channel dim

            if idx % self.eval_save_example_every == 0:
                img = self.restore_transform(img)  # Un-normalise normalised image
                gt_count = gt.sum() / self.cfg_data.LABEL_FACTOR  # Divide to get actual non-scaled count
                # gt_count = torch.round(gt_count)  # People's density can be outside image, thus not being an integer.

                plt.imshow(img)  # No displayed on screen. Just to save the image in corresponding folder.
                save_path = os.path.join(self.cfg.PICS_DIR, f'img_{idx}.jpg')
                plt.title(f'GT count: {gt_count:.3f}')
                plt.savefig(save_path)

                plt.imshow(gt, cmap=CM.jet)
                save_path = os.path.join(self.cfg.PICS_DIR, f'gt_{idx}.jpg')
                plt.title(f'GT count: {gt_count:.3f}')
                plt.savefig(save_path)

        # Images in val loader might not be in order. Save a mapping from image index to actual image path.
        idx_to_img_path = os.path.join(os.path.join(self.cfg.PICS_DIR, 'idx_to_img_path.csv'))
        data_files = self.val_loader.dataset.data_files
        with open(idx_to_img_path, 'w') as f:
            write = csv.writer(f)
            write.writerows(list(zip(np.arange(len(data_files)), data_files)))  # each element is (idx, img_path)

    def save_state(self, name_extra=''):
        """ Saves the variables needed to continue training later. """

        if name_extra:  # Sometimes, we want to manually add some extra info. E.g. when new best MAE
            save_name = f'{self.cfg.STATE_DICTS_DIR}/save_state_ep_{self.epoch}_{name_extra}.pth'
        else:
            save_name = f'{self.cfg.STATE_DICTS_DIR}/save_state_ep_{self.epoch}.pth'

        save_sate = {
            'epoch': self.epoch,  # Current epoch
            'best_epoch': self.best_epoch,  # Epoch where we got the best MAE
            'best_mae': self.best_mae,  # Best MAE so far
            'net': self.model.state_dict(),  # The entire network
            'optim': self.optim.state_dict(),  # The optimiser used to train the model. Is needed for Adam momentum etc.
            'scheduler': self.scheduler.state_dict(),  # Learning rate scheduler
            'save_dir_path': self.cfg.SAVE_DIR,  # Where to save evaluation predictions, save state, etc.
        }

        torch.save(save_sate, save_name)

    def load_state(self, state_path):  # Not supported yet!
        """ Loads the variables to continue training. """

        resume_state = torch.load(state_path)
        self.epoch = resume_state['epoch']
        self.best_epoch = resume_state['best_epoch']
        self.best_mae = resume_state['best_mae']

        self.model.load_state_dict(resume_state['net'])
        self.optim.load_state_dict(resume_state['optim'])
        self.scheduler.load_state_dict(resume_state['scheduler'])


def print_fancy_new_best_MAE():
    """ For that extra bit of dopamine rush when you get a new high-score. """

    new_best = '#' + '=' * 20 + '<' * 3 + ' NEW BEST MAE ' + '>' * 3 + '=' * 20 + '#'
    n_chars = len(new_best)
    bar = '#' + '=' * (n_chars - 2) + '#'
    print(bar)
    print(new_best)
    print(bar)
