import torch

from rsl_rl.algorithms.ppo_amp import (
    _fill_non_overlapping_grid_regions,
    _rasterize_rectangles,
    _replace_grid_history,
)


def test_rasterize_axis_aligned_rectangle() -> None:
    grid = _rasterize_rectangles(
        center_x=torch.tensor([0.0]),
        center_y=torch.tensor([0.0]),
        yaw=torch.tensor([0.0]),
        length=torch.tensor([0.2]),
        width=torch.tensor([0.4]),
        local_size=(1.0, 1.0),
        resolution=0.1,
    ).view(1, 10, 10)

    assert grid.sum().item() == 8
    assert torch.all(grid[0, 4:6, 3:7] == 1.0)


def test_rasterize_rotated_rectangle_swaps_extents() -> None:
    grid = _rasterize_rectangles(
        center_x=torch.tensor([0.0]),
        center_y=torch.tensor([0.0]),
        yaw=torch.tensor([torch.pi / 2]),
        length=torch.tensor([0.2]),
        width=torch.tensor([0.4]),
        local_size=(1.0, 1.0),
        resolution=0.1,
    ).view(1, 10, 10)

    assert grid.sum().item() == 8
    assert torch.all(grid[0, 3:7, 4:6] == 1.0)


def test_random_regions_respect_forbidden_cells_and_density() -> None:
    torch.manual_seed(7)
    grid = torch.zeros(64, 16, 10)
    forbidden = torch.zeros_like(grid, dtype=torch.bool)
    forbidden[:, 7:9, 4:6] = True

    result = _fill_non_overlapping_grid_regions(
        grid,
        density_range=(0.06, 0.08),
        rectangle_rows_range=(1, 2),
        rectangle_cols_range=(1, 4),
        forbidden=forbidden,
    )

    occupied = result.sum(dim=(1, 2))
    assert torch.all(result[forbidden] == 0.0)
    assert torch.all(occupied >= 10)
    # A final 2 x 4-cell rectangle can overshoot the 13-cell target.
    assert torch.all(occupied <= 20)
    assert torch.all((result == 0.0) | (result == 1.0))


def test_replace_grid_history_preserves_non_grid_observations() -> None:
    observations = torch.arange(26, dtype=torch.float32).reshape(2, 13)
    prefix = observations[:, :5].clone()
    grid = torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])

    _replace_grid_history(observations, grid, history_length=2)

    assert torch.equal(observations[:, :5], prefix)
    assert torch.equal(observations[:, 5:], grid.repeat(1, 2))
