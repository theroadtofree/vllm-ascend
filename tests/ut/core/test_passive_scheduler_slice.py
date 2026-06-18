# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PassiveScheduler layer-slice boundary computation."""

import pytest

from vllm_ascend.core.passive_scheduler import PassiveScheduler


class TestComputeSliceBoundaries:
    """Tests for PassiveScheduler._compute_slice_boundaries."""

    @staticmethod
    def _sizes(boundaries: list[tuple[int, int]]) -> list[int]:
        return [end - start for start, end in boundaries]

    def test_example_62_layers_5_slices(self):
        """62 local layers / 5 slices -> [13, 13, 12, 12, 12]."""
        boundaries = PassiveScheduler._compute_slice_boundaries(62, 5)
        sizes = self._sizes(boundaries)
        assert sizes == [13, 13, 12, 12, 12]
        assert sum(sizes) == 62

    def test_example_60_layers_5_slices(self):
        """60 local layers / 5 slices -> [12, 12, 12, 12, 12]."""
        boundaries = PassiveScheduler._compute_slice_boundaries(60, 5)
        sizes = self._sizes(boundaries)
        assert sizes == [12, 12, 12, 12, 12]
        assert sum(sizes) == 60

    def test_example_61_layers_5_slices(self):
        """61 local layers / 5 slices -> [13, 12, 12, 12, 12]."""
        boundaries = PassiveScheduler._compute_slice_boundaries(61, 5)
        sizes = self._sizes(boundaries)
        assert sizes == [13, 12, 12, 12, 12]
        assert sum(sizes) == 61

    def test_example_63_layers_5_slices(self):
        """63 local layers / 5 slices -> [13, 13, 13, 12, 12]."""
        boundaries = PassiveScheduler._compute_slice_boundaries(63, 5)
        sizes = self._sizes(boundaries)
        assert sizes == [13, 13, 13, 12, 12]
        assert sum(sizes) == 63

    def test_single_slice(self):
        """All layers in one slice."""
        boundaries = PassiveScheduler._compute_slice_boundaries(62, 1)
        sizes = self._sizes(boundaries)
        assert sizes == [62]
        assert sum(sizes) == 62

    def test_each_layer_one_slice(self):
        """62 layers / 62 slices -> 62 slices of size 1."""
        boundaries = PassiveScheduler._compute_slice_boundaries(62, 62)
        sizes = self._sizes(boundaries)
        assert sizes == [1] * 62
        assert sum(sizes) == 62

    def test_more_slices_than_layers(self):
        """5 layers / 10 slices -> first 5 slices size 1, rest size 0."""
        boundaries = PassiveScheduler._compute_slice_boundaries(5, 10)
        sizes = self._sizes(boundaries)
        assert sizes == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        assert sum(sizes) == 5

    def test_zero_layers(self):
        """Zero local layers -> empty boundaries."""
        boundaries = PassiveScheduler._compute_slice_boundaries(0, 5)
        assert boundaries == []

    def test_zero_slices(self):
        """Zero slices -> empty boundaries."""
        boundaries = PassiveScheduler._compute_slice_boundaries(62, 0)
        assert boundaries == []

    def test_size_difference_at_most_one(self):
        """For any valid input, max(slice_size) - min(slice_size) <= 1."""
        for num_layers in [1, 7, 13, 29, 62, 100]:
            for num_slices in [1, 2, 3, 5, 7, 10, 13, num_layers, num_layers + 5]:
                boundaries = PassiveScheduler._compute_slice_boundaries(
                    num_layers, num_slices
                )
                if not boundaries:
                    continue
                sizes = self._sizes(boundaries)
                assert max(sizes) - min(sizes) <= 1, (
                    f"num_layers={num_layers}, num_slices={num_slices}, "
                    f"sizes={sizes}"
                )
                assert sum(sizes) == num_layers, (
                    f"sum mismatch: {sum(sizes)} != {num_layers}"
                )

    def test_larger_slices_come_first(self):
        """Slice sizes must be non-increasing."""
        for num_layers in [1, 10, 50, 99]:
            for num_slices in [1, 3, 7, 11, 20]:
                boundaries = PassiveScheduler._compute_slice_boundaries(
                    num_layers, num_slices
                )
                if not boundaries:
                    continue
                sizes = self._sizes(boundaries)
                for i in range(len(sizes) - 1):
                    assert sizes[i] >= sizes[i + 1], (
                        f"sizes not non-increasing: {sizes}"
                    )


class TestResolveSliceCount:
    """Tests for PassiveScheduler._resolve_slice_count with YAML config."""

    @pytest.fixture
    def scheduler_with_config(self):
        """Build a minimal PassiveScheduler-like object with a YAML config."""
        # We only need the _resolve_slice_count method; avoid real init.
        class FakeScheduler:
            _layer_slice_config = {
                16: 24,
                8: 10,
                4: 4,
                3: 6,
                2: 5,
                1: 5,
                0: 5,
            }

        # Bind the real method to the fake instance
        FakeScheduler._resolve_slice_count = PassiveScheduler._resolve_slice_count
        return FakeScheduler()

    def test_yaml_16k_tokens(self, scheduler_with_config):
        """16000 tokens >= 16k threshold -> 24 slices."""
        assert scheduler_with_config._resolve_slice_count(16000) == 24
        assert scheduler_with_config._resolve_slice_count(20000) == 24

    def test_yaml_8k_tokens(self, scheduler_with_config):
        """8000 tokens >= 8k but < 16k -> 10 slices."""
        assert scheduler_with_config._resolve_slice_count(8000) == 10
        assert scheduler_with_config._resolve_slice_count(12000) == 10

    def test_yaml_4k_tokens(self, scheduler_with_config):
        """4000 tokens >= 4k but < 8k -> 4 slices."""
        assert scheduler_with_config._resolve_slice_count(4000) == 4
        assert scheduler_with_config._resolve_slice_count(7000) == 4

    def test_yaml_3k_tokens(self, scheduler_with_config):
        """3000 tokens >= 3k but < 4k -> 6 slices."""
        assert scheduler_with_config._resolve_slice_count(3000) == 6
        assert scheduler_with_config._resolve_slice_count(3999) == 6

    def test_yaml_1k_tokens(self, scheduler_with_config):
        """1000 tokens >= 1k but < 2k -> 5 slices."""
        assert scheduler_with_config._resolve_slice_count(1000) == 5
        assert scheduler_with_config._resolve_slice_count(1500) == 5

    def test_yaml_0_tokens(self, scheduler_with_config):
        """0 tokens >= 0k -> 5 slices (fallback inside YAML)."""
        assert scheduler_with_config._resolve_slice_count(0) == 5

    def test_no_config(self):
        """No YAML config -> 0 slices (disabled)."""
        class FakeScheduler:
            _layer_slice_config = None

        FakeScheduler._resolve_slice_count = PassiveScheduler._resolve_slice_count
        sched = FakeScheduler()
        assert sched._resolve_slice_count(10000) == 0
