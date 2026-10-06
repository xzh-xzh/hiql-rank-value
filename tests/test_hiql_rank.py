"""Synthetic ranking contracts; use JAX_PLATFORMS=cuda for CUDA validation."""

import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'impls'))

import flax
import jax
import jax.numpy as jnp
import numpy as np

from agents.hiql import HIQLAgent, get_config
from utils.datasets import Dataset, HGCDataset


def make_dataset(config):
    ids = np.arange(9, dtype=np.float32)
    terminals = np.isin(ids, [3, 6, 8]).astype(np.float32)
    return HGCDataset(Dataset.create(
        observations=np.stack([ids, 1 + ids / 10], axis=-1),
        actions=ids[:, None] / 10,
        valids=1 - terminals,
        terminals=np.minimum(terminals + np.r_[terminals[1:], 1], 1),
    ), config)


def small_config(mode='rank'):
    config = get_config()
    config.value_loss_type = mode
    config.actor_hidden_dims = (16, 16)
    config.value_hidden_dims = (16, 16)
    config.rep_dim = 4
    return config


class TableNetwork:
    def select(self, name):
        assert name == 'value', 'Ranking must not read a target network'

        def apply(observations, goals, params=None):
            return params[:, observations[:, 0].astype(jnp.int32)]

        return apply


class RankTests(unittest.TestCase):
    def test_sampling_boundaries_and_goal_endpoint(self):
        dataset = make_dataset(small_config())
        idxs = np.tile(dataset.dataset.valid_idxs, 128)
        np.random.seed(10)
        batch = dataset.sample(len(idxs), idxs=idxs)
        later = batch['rank_observations'][:, 0].astype(int)
        goals = batch['value_goals'][:, 0].astype(int)
        valid = batch['rank_masks'].astype(bool)
        ends = dataset.terminal_locs[np.searchsorted(dataset.terminal_locs, idxs)]
        self.assertTrue(np.all(idxs[valid] < later[valid]))
        self.assertTrue(np.all(later <= goals))
        self.assertTrue(np.all(goals <= ends))
        self.assertTrue(np.any((later == goals) & valid))
        self.assertTrue(np.any(~valid))
        np.testing.assert_array_equal(valid, idxs < ends)
        np.testing.assert_array_equal(batch['observations'][:, 0], idxs)
        np.testing.assert_array_equal(batch['low_actor_goals'][:, 0], np.minimum(idxs + 25, ends))

    def test_both_endpoints_receive_correct_gradients(self):
        agent = HIQLAgent(rng=jax.random.PRNGKey(0), network=TableNetwork(),
                          config=flax.core.FrozenDict(value_loss_type='rank', rank_margin=0.1))
        batch = dict(observations=jnp.array([[0.0]]), rank_observations=jnp.array([[1.0]]),
                     value_goals=jnp.array([[1.0]]), rank_masks=jnp.ones(1))
        table = jnp.zeros((2, 2))
        loss, grads = jax.value_and_grad(lambda p: agent.value_loss(batch, p)[0])(table)
        self.assertAlmostEqual(float(loss), 0.2, places=6)
        np.testing.assert_array_equal(grads, [[1, -1], [1, -1]])
        good = table.at[:, 1].set(0.2)
        self.assertEqual(float(agent.value_loss(batch, good)[0]), 0)
        shifted = good + 37
        self.assertEqual(float(agent.value_loss(batch, shifted)[0]), 0)
        batch['rank_masks'] = jnp.zeros(1)
        loss, grads = jax.value_and_grad(lambda p: agent.value_loss(batch, p)[0])(table)
        self.assertEqual(float(loss), 0)
        np.testing.assert_array_equal(grads, 0)

    def test_rank_update_changes_value_without_bellman_targets(self):
        config = small_config()
        batch = jax.device_put(make_dataset(config).sample(32))
        agent = HIQLAgent.create(0, batch['observations'][:1], batch['actions'][:1], config)
        altered = dict(batch, rewards=jnp.full(32, 9999.0), masks=jnp.zeros(32),
                       next_observations=jnp.full_like(batch['next_observations'], -9999))
        np.testing.assert_allclose(agent.value_loss(batch, agent.network.params)[0],
                                   agent.value_loss(altered, agent.network.params)[0])
        (loss, _), grads = jax.value_and_grad(agent.total_loss, argnums=1, has_aux=True)(batch, agent.network.params)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(np.isfinite(g).all() for g in jax.tree_util.tree_leaves(grads)))
        updated, _ = agent.update(batch)
        for before, after in zip(jax.tree_util.tree_leaves(agent.network.params['modules_target_value']),
                                 jax.tree_util.tree_leaves(updated.network.params['modules_target_value'])):
            np.testing.assert_array_equal(before, after)
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(
            jax.tree_util.tree_leaves(agent.network.params['modules_value']),
            jax.tree_util.tree_leaves(updated.network.params['modules_value']))))

    def test_frame_stack_and_shared_image_augmentation(self):
        config = small_config()
        config.frame_stack = 2
        config.p_aug = 1.0
        ids = np.arange(6, dtype=np.uint8)
        ramp = np.arange(36, dtype=np.uint8).reshape(6, 6, 1)
        observations = np.stack([ramp + i for i in ids])
        dataset = HGCDataset(Dataset.create(
            observations=observations, actions=np.zeros((6, 1)),
            terminals=np.array([0, 0, 0, 0, 1, 1]), valids=np.array([1, 1, 1, 1, 1, 0]),
        ), config)
        # Only one later state: j=k, so both must receive the same random crop.
        batch = dataset.sample(8, idxs=np.full(8, 3))
        np.testing.assert_array_equal(batch['rank_observations'], batch['value_goals'])
        self.assertEqual(batch['rank_observations'].shape, (8, 6, 6, 2))

    def test_rejects_invalid_configuration(self):
        config = small_config()
        for margin in [0.0, -1.0, float('nan')]:
            config.rank_margin = margin
            with self.assertRaises(ValueError):
                HIQLAgent.create(0, np.ones((1, 2)), np.ones((1, 1)), config)

    def test_upper_hinge_preserves_ranking_force_and_pulls_back_large_gaps(self):
        batch = dict(observations=jnp.array([[0.0]]), rank_observations=jnp.array([[1.0]]),
                     value_goals=jnp.array([[1.0]]), rank_masks=jnp.ones(1))
        for ratio in (0.5, 0.8):
            for weight in (0.05, 0.1, 0.2):
                agent = HIQLAgent(rng=jax.random.PRNGKey(0), network=TableNetwork(),
                    config=flax.core.FrozenDict(value_loss_type='rank', rank_margin=3.0,
                        rank_upper_margin_ratio=ratio, rank_upper_weight=weight))
                for gap, slope in ((0.0, -1.0), ((3.0 * ratio + 3.0) / 2, -1 + weight),
                                   (4.0, weight)):
                    table = jnp.array([[0.0, gap], [0.0, gap]])
                    loss, grads = jax.value_and_grad(lambda p: agent.value_loss(batch, p)[0])(table)
                    np.testing.assert_allclose(grads, [[-slope, slope]] * 2, atol=1e-6)
                    shifted_loss = agent.value_loss(batch, table + 16)[0]
                    np.testing.assert_allclose(loss, shifted_loss, atol=3e-6)
                masked = dict(batch, rank_masks=jnp.zeros(1))
                loss, grads = jax.value_and_grad(lambda p: agent.value_loss(masked, p)[0])(table)
                self.assertEqual(float(loss), 0)
                np.testing.assert_array_equal(grads, 0)

    def test_upper_hinge_constrains_each_value_head(self):
        agent = HIQLAgent(rng=jax.random.PRNGKey(0), network=TableNetwork(),
            config=flax.core.FrozenDict(value_loss_type='rank', rank_margin=3.0,
                rank_upper_margin_ratio=0.5, rank_upper_weight=0.1))
        batch = dict(observations=jnp.array([[0.0]]), rank_observations=jnp.array([[1.0]]),
                     value_goals=jnp.array([[1.0]]), rank_masks=jnp.ones(1))
        # Averaging heads first would hide the oversized second head.
        table = jnp.array([[0., -1.], [0., 7.]])
        loss, grads = jax.value_and_grad(lambda p: agent.value_loss(batch, p)[0])(table)
        self.assertAlmostEqual(float(loss), 4.55, places=5)
        np.testing.assert_allclose(grads, [[1, -1], [-0.1, 0.1]], atol=1e-6)

    def test_upper_hinge_rejects_nonfinite_or_invalid_settings(self):
        for name, values in [('rank_margin', [float('inf')]),
                             ('rank_upper_weight', [-0.1, float('nan'), float('inf')]),
                             ('rank_upper_margin_ratio', [0., 1., -0.1, float('nan'), float('inf')])]:
            for value in values:
                config = small_config()
                config[name] = value
                with self.assertRaises(ValueError):
                    HIQLAgent.create(0, np.ones((1, 2)), np.ones((1, 1)), config)

    def test_td_and_actors_match_original_code(self):
        reference = ROOT / 'tests' / 'reference_hiql.py'
        if not reference.exists():
            self.skipTest('Set up reference_hiql.py from the unmodified Git HEAD for the regression check')
        spec = importlib.util.spec_from_file_location('reference_hiql', reference)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        config = small_config('td')
        batch = jax.device_put(make_dataset(config).sample(16))
        current = HIQLAgent.create(0, batch['observations'][:1], batch['actions'][:1], config)
        baseline = original.HIQLAgent.create(0, batch['observations'][:1], batch['actions'][:1], config)
        for method in ['value_loss', 'low_actor_loss', 'high_actor_loss']:
            np.testing.assert_allclose(getattr(current, method)(batch, current.network.params)[0],
                                       getattr(baseline, method)(batch, baseline.network.params)[0], rtol=1e-6)
        updated, _ = current.update(batch)
        baseline_updated, _ = baseline.update(batch)
        for a, b in zip(jax.tree_util.tree_leaves(updated.network.params),
                        jax.tree_util.tree_leaves(baseline_updated.network.params)):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)


if __name__ == '__main__':
    print(f'Test backend: {jax.default_backend()}')
    unittest.main(verbosity=2)
