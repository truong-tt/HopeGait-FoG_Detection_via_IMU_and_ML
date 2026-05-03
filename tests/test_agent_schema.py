"""Lock the contract between agent/ output and src/data_pipeline/dataset.py.

The agent writes `data/synthetic/win_<W>/subj_S<id>_synth_x.npy` (N, T, 9)
float32 plus `_y.npy` (N, T) int64. dataset.py's LOSO loader parses the
subject ID from the filename and concatenates per-subject windows. If either
side renames files, changes shapes, or shifts dtypes, this test fails fast
instead of at the next CI smoke run.

This test does NOT import agent/hopegait_agent.py — that module hard-exits
on missing google-genai. We reproduce the file layout the agent emits and
verify dataset.py consumes it cleanly.
"""

import os
import tempfile

import numpy as np
import pytest


WIN = 128
N_CHANNELS = 9


def _write_subject(out_dir, sid, n_windows, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_windows, WIN, N_CHANNELS)).astype(np.float32)
    # Half-positive last-step labels so the LOSO meta computes a sane pos_rate.
    y = (rng.random((n_windows, WIN)) < 0.5).astype(np.int64)
    np.save(os.path.join(out_dir, f'subj_S{sid:02d}_synth_x.npy'), x)
    np.save(os.path.join(out_dir, f'subj_S{sid:02d}_synth_y.npy'), y)


def test_agent_layout_loads_through_loso():
    """File names + shapes from agent.window_and_save must satisfy dataset.py."""
    from data_pipeline.dataset import (
        create_loso_dataloaders,
        get_all_subjects,
    )

    with tempfile.TemporaryDirectory() as work:
        win_dir = os.path.join(work, f'win_{WIN}')
        os.makedirs(win_dir)
        for i, sid in enumerate((1, 2, 3)):
            _write_subject(win_dir, sid, n_windows=8, seed=i)

        subjects = get_all_subjects(win_dir)
        assert subjects == ['S01', 'S02', 'S03'], subjects

        train_loader, val_loader, test_loader, scaler, meta = create_loso_dataloaders(
            win_dir, test_subject='S03', batch_size=4,
            augment_train=False, num_workers=0, seed=0,
        )

        assert meta['test_subject'] == 'S03'
        assert meta['val_subject'] in ('S01', 'S02')
        assert meta['train_subjects'] and meta['test_subject'] not in meta['train_subjects']
        assert meta['train_windows'] > 0
        assert meta['test_windows'] == 8

        x_batch, y_batch = next(iter(train_loader))
        assert x_batch.shape[1:] == (WIN, N_CHANNELS), x_batch.shape
        assert y_batch.shape[1:] == (WIN,), y_batch.shape
        assert x_batch.dtype.is_floating_point
        assert not y_batch.dtype.is_floating_point

        x_test, y_test = next(iter(test_loader))
        assert x_test.shape[1:] == (WIN, N_CHANNELS)
        assert y_test.shape[1:] == (WIN,)


def test_agent_signal_synthesizer_returns_expected_shapes():
    """synth_signal.synthesize_subject keeps its (n, 6) + (n,) contract."""
    import sys
    agent_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'agent')
    )
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)
    import synth_signal

    fs = 64.0
    duration_s = 4.0
    profile = {
        'subject_id': 1, 'gait_freq_hz': 1.8,
        'tremor_band_hz': [4.5, 6.0], 'fog_severity': 12.0, 'tremor_amp_g': 0.05,
    }
    events = [{'type': 'fog', 'start_s': 1.0, 'duration_s': 1.5}]
    signal, labels = synth_signal.synthesize_subject(
        profile, events, duration_s=duration_s, fs=fs, seed=0,
    )

    n = int(round(duration_s * fs))
    assert signal.shape == (n, 6)
    assert labels.shape == (n,)
    assert signal.dtype == np.float32
    assert labels.dtype == np.int64
    # Some FoG samples must be labelled 1; gait baseline must remain 0 elsewhere.
    assert labels.sum() > 0
    assert labels.sum() < n
