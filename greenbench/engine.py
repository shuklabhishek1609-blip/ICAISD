"""BaseEngine subclass that records cost alongside accuracy.

Two things the stock LargeST engine will not give us:

  1. Training energy. We meter each epoch so early stopping is priced in --
     a model that needs 80 epochs to converge is genuinely more expensive
     than one that needs 20, and total-run energy is the only metric that
     captures that.
  2. Test metrics as data. BaseEngine.evaluate('test') writes to a log and
     returns None, so we re-implement it to return the per-horizon arrays.

Everything else -- masking, inverse transform, checkpointing -- is inherited
unchanged, so our numbers stay comparable to published LargeST results.
"""

import time

import numpy as np
import torch

from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_rmse, compute_all_metrics

from .instrument import GPUEnergyMeter


class InstrumentedEngine(BaseEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_joules = 0.0
        self.train_seconds = 0.0
        self.epochs_run = 0
        self.epoch_joules = []
        self.epoch_seconds = []
        self.test_results = None
        self._meter = GPUEnergyMeter()

    def train(self):
        self._logger.info('Start training (instrumented)!')

        # Historical Last has no trainable parameters, so there is nothing to
        # back-propagate and no training energy to bill it. Rather than fake a
        # gradient to keep the loop uniform, skip it and record a true zero --
        # "this baseline costs nothing to fit" is a fact the paper wants to
        # state, not an artefact to paper over. Still checkpoint, because
        # evaluate_test reloads from disk like every other model.
        if not any(p.requires_grad for p in self.model.parameters()):
            self._logger.info('No trainable parameters -- skipping training, '
                              'train energy recorded as 0 J.')
            self.save_model(self._save_path)
            self.train_joules = 0.0
            self.train_seconds = 0.0
            self.epochs_run = 0
            self.test_results = self.evaluate_test()
            return self.test_results

        wait = 0
        min_loss = np.inf
        run_t0 = time.time()

        for epoch in range(self._max_epochs):
            self._meter.start()
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse = self.train_batch()
            t2 = time.time()
            epoch_j = self._meter.stop()

            self.epoch_joules.append(epoch_j)
            self.epoch_seconds.append(t2 - t1)
            self.epochs_run = epoch + 1

            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = self.evaluate('val')
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            message = ('Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, '
                       'Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, '
                       'Valid MAPE: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, '
                       'Train Energy: {:.1f}J, LR: {:.4e}')
            self._logger.info(message.format(
                epoch + 1, mtrain_loss, mtrain_rmse, mtrain_mape,
                mvalid_loss, mvalid_rmse, mvalid_mape,
                (t2 - t1), (v2 - v1), epoch_j, cur_lr))

            if mvalid_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info('Val loss decrease from {:.4f} to {:.4f}'.format(min_loss, mvalid_loss))
                min_loss = mvalid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch + 1, min_loss))
                    break

        self.train_seconds = time.time() - run_t0
        # Sum of per-epoch readings rather than one measurement across the
        # whole run: validation passes sit between epochs and should not be
        # billed to training. nansum of an all-NaN list is 0.0, which would
        # report an unmetered run as a free one -- keep it NaN so a missing
        # measurement stays visibly missing.
        self.train_joules = (float(np.nansum(self.epoch_joules))
                             if np.any(np.isfinite(self.epoch_joules))
                             else float('nan'))
        self._logger.info('Total training energy: {:.1f}J over {} epochs ({:.4f} kWh)'.format(
            self.train_joules, self.epochs_run, self.train_joules / 3.6e6))

        self.test_results = self.evaluate_test()
        return self.test_results

    def evaluate_test(self):
        """Like BaseEngine.evaluate('test') but returns the numbers."""
        self.load_model(self._save_path)
        self.model.eval()

        preds, labels = [], []
        with torch.no_grad():
            for X, label in self._dataloader['test_loader'].get_iterator():
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, label)
                pred, label = self._inverse_transform([pred, label])
                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        h_mae, h_mape, h_rmse = [], [], []
        for i in range(self.model.horizon):
            mae, mape, rmse = compute_all_metrics(preds[:, i, :], labels[:, i, :], mask_value)
            self._logger.info('Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'.format(
                i + 1, mae, rmse, mape))
            h_mae.append(mae)
            h_mape.append(mape)
            h_rmse.append(rmse)

        self._logger.info('Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'.format(
            np.mean(h_mae), np.mean(h_rmse), np.mean(h_mape)))

        return {
            'mae': float(np.mean(h_mae)),
            'rmse': float(np.mean(h_rmse)),
            'mape': float(np.mean(h_mape)),
            'horizon_mae': [float(v) for v in h_mae],
            'horizon_rmse': [float(v) for v in h_rmse],
            'horizon_mape': [float(v) for v in h_mape],
        }

    def measure_inference_energy(self, n_batches=30, min_seconds=5.0):
        """Energy per 1000 test samples at inference.

        Deployment cost, as distinct from training cost -- a traffic centre
        trains once and then infers every five minutes forever, so this is
        the number that actually compounds.

        A fixed batch count is not enough at the cheap end of the lineup. Thirty
        batches of NLinear is roughly 10 ms of GPU work, well under what the
        NVML energy counter resolves, so it returns a flat 0 J -- and "this
        model costs no energy to run" is precisely the overclaim this paper
        must not make. So replay the batches until a wall-clock floor has
        passed and divide by the samples actually processed.
        """
        self.model.eval()
        loader = self._dataloader['test_loader']
        meter = GPUEnergyMeter()

        batches = []
        for i, (X, label) in enumerate(loader.get_iterator()):
            if i >= n_batches:
                break
            batches.append(self._to_device(self._to_tensor([X, label])))

        if not batches:
            return float('nan')

        with torch.no_grad():  # warmup
            for X, label in batches[:3]:
                self.model(X, label)

        meter.start()
        samples, passes = 0, 0
        t0 = time.time()
        with torch.no_grad():
            while True:
                for X, label in batches:
                    self.model(X, label)
                    samples += X.shape[0]
                # Without this the CPU races ahead queueing kernels and the
                # elapsed-time test exits before the GPU has done the work.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                passes += 1
                if time.time() - t0 >= min_seconds:
                    break
        joules = meter.stop()

        self._logger.info(
            'Inference energy: {:.1f}J over {} samples ({} passes, {:.1f}s)'.format(
                joules, samples, passes, time.time() - t0))

        return joules / samples * 1000 if samples else float('nan')
