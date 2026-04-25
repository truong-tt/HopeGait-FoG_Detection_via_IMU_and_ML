"""LOSO splitter and Dataset for windowed IMU features.

`x` is shape (T, 9): linear_acc(0..2) | gravity(3..5) | gyro(6..8).
`y` is shape (T,) per-timestep — last entry is the real-time label the MCU
predicts; the full vector trains the dense auxiliary head.
"""

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from .dsp import RobustScaler
except ImportError:
    from dsp import RobustScaler


# Channel layout — keep in sync with preprocess.py.
ACC_SLICE = slice(0, 3)
GRAV_SLICE = slice(3, 6)
GYRO_SLICE = slice(6, 9)


def _random_rotation_matrix(max_angle_rad, rng):
    # Uniform random axis on the unit sphere, angle in [-max, +max].
    axis = rng.normal(size=3)
    n = np.linalg.norm(axis)
    if n < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis / n
    angle = rng.uniform(-max_angle_rad, max_angle_rad)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return R.astype(np.float32)


def _apply_rotation(window, R):
    """Apply rotation R (3x3) jointly to acc, gravity, gyro slices.

    All three vectors live in the same body frame, so they share a single R.
    Acts on a (T, 9) array in place-safe fashion (returns new array).
    """
    out = window.copy()
    out[:, ACC_SLICE] = window[:, ACC_SLICE] @ R.T
    out[:, GRAV_SLICE] = window[:, GRAV_SLICE] @ R.T
    out[:, GYRO_SLICE] = window[:, GYRO_SLICE] @ R.T
    return out


class FoGDataset(Dataset):
    def __init__(self, features, labels, augment=False, augment_prob=0.5,
                 jitter_sigma=0.03, scale_sigma=0.05, max_time_shift=5,
                 rotation_max_deg=15.0, rotation_prob=0.5):
        self.X = np.ascontiguousarray(features, dtype=np.float32)
        # Accept (N, T) dense labels OR legacy 1D (N,) last-step labels.
        labels = np.asarray(labels)
        if labels.ndim == 1:
            T = self.X.shape[1]
            labels = np.broadcast_to(labels[:, None], (labels.shape[0], T)).copy()
        self.y = labels.astype(np.int64)
        self.augment = augment
        self.p = augment_prob
        self.jitter_sigma = jitter_sigma
        self.scale_sigma = scale_sigma
        self.max_time_shift = max_time_shift
        self.rotation_max_rad = np.deg2rad(rotation_max_deg)
        self.rotation_prob = rotation_prob

    def __len__(self):
        return len(self.y)

    def _jitter(self, x, rng):
        return x + rng.normal(0, self.jitter_sigma, x.shape).astype(np.float32)

    def _scaling(self, x, rng):
        factor = rng.normal(1.0, self.scale_sigma, (1, x.shape[1])).astype(np.float32)
        return x * factor

    def _time_shift(self, x, y, rng):
        k = int(rng.integers(-self.max_time_shift, self.max_time_shift + 1))
        if k == 0:
            return x, y
        # Roll labels with the signal so dense supervision stays aligned.
        return np.roll(x, shift=k, axis=0), np.roll(y, shift=k, axis=0)

    def _rotate(self, x, rng):
        R = _random_rotation_matrix(self.rotation_max_rad, rng)
        return _apply_rotation(x, R)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.augment:
            rng = np.random.default_rng()
            if rng.random() < self.rotation_prob:
                x = self._rotate(x, rng)
            if rng.random() < self.p:
                x = self._jitter(x, rng)
            if rng.random() < self.p:
                x = self._scaling(x, rng)
            if rng.random() < self.p:
                x, y = self._time_shift(x, y, rng)
        x_t = torch.from_numpy(np.ascontiguousarray(x))
        y_t = torch.from_numpy(np.ascontiguousarray(y))
        return x_t, y_t


def _subject_id_from_filename(path):
    # Parse subject ID from 'subj_<ID>_<FILEID>_x.npy'. Grouping by the full
    # filename (old behavior) put a single patient's recordings in different
    # folds, leaking identity across train/test.
    base = os.path.basename(path).replace('_x.npy', '')
    parts = base.split('_')
    if len(parts) < 2 or parts[0] != 'subj':
        raise ValueError(f"Unexpected filename format: {path}")
    return parts[1]


def _group_files_by_subject(data_dir):
    x_files = sorted(glob.glob(os.path.join(data_dir, '*_x.npy')))
    groups = {}
    for xf in x_files:
        groups.setdefault(_subject_id_from_filename(xf), []).append(xf)
    return groups


def get_all_subjects(data_dir):
    return sorted(_group_files_by_subject(data_dir).keys())


def _load_files(x_files):
    xs, ys = [], []
    for xf in x_files:
        yf = xf.replace('_x.npy', '_y.npy')
        if not os.path.exists(yf):
            continue
        xs.append(np.load(xf))
        ys.append(np.load(yf))
    if not xs:
        return None, None
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _last_step(y):
    return y[:, -1] if y.ndim == 2 else y


def create_loso_dataloaders(data_dir, test_subject, val_subject=None, batch_size=64,
                            scaler=None, augment_train=True, num_workers=0, seed=42,
                            rotation_max_deg=15.0, rotation_prob=0.5):
    """Returns (train_loader, val_loader, test_loader, scaler, meta).

    Picks an inner val subject from the training pool so model selection
    never peeks at the test subject. Scaler is fit on training data only,
    unless one is supplied (eval path).
    """
    groups = _group_files_by_subject(data_dir)
    if test_subject not in groups:
        raise ValueError(f"Test subject '{test_subject}' not found in {data_dir}.")

    all_subjects = sorted(groups.keys())
    non_test = [s for s in all_subjects if s != test_subject]
    if not non_test:
        raise ValueError("Need at least 2 subjects for LOSO.")

    if val_subject is None:
        rng = random.Random(f"{seed}:{test_subject}")
        val_subject = rng.choice(non_test) if len(non_test) > 1 else non_test[0]
    if val_subject == test_subject:
        raise ValueError("val_subject must differ from test_subject.")
    if val_subject not in groups:
        raise ValueError(f"Validation subject '{val_subject}' not found in {data_dir}.")

    train_subjects = [s for s in non_test if s != val_subject]

    train_files = [f for s in train_subjects for f in groups[s]]
    X_train, y_train = _load_files(train_files)
    X_val, y_val = _load_files(groups[val_subject])
    X_test, y_test = _load_files(groups[test_subject])

    if X_train is None:
        raise ValueError("Training pool loaded zero windows.")

    if scaler is None:
        scaler = RobustScaler().fit(X_train)

    X_train = scaler.transform(X_train)
    if X_val is not None:
        X_val = scaler.transform(X_val)
    if X_test is not None:
        X_test = scaler.transform(X_test)

    train_ds = FoGDataset(X_train, y_train, augment=augment_train,
                          rotation_max_deg=rotation_max_deg, rotation_prob=rotation_prob)
    val_ds = FoGDataset(X_val, y_val, augment=False) if X_val is not None else None
    test_ds = FoGDataset(X_test, y_test, augment=False) if X_test is not None else None

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers) if val_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers) if test_ds is not None else None

    meta = {
        'test_subject': test_subject,
        'val_subject': val_subject,
        'train_subjects': train_subjects,
        'train_windows': int(len(y_train)),
        'val_windows': int(len(y_val)) if y_val is not None else 0,
        'test_windows': int(len(y_test)) if y_test is not None else 0,
        'train_pos_rate': float(np.mean(_last_step(y_train))),
    }
    return train_loader, val_loader, test_loader, scaler, meta
