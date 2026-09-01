import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPOAMP
from rsl_rl.modules import ActorCriticFoothold


def test_foothold_actor_shapes() -> None:
    obs = TensorDict(
        {
            "policy": torch.randn(8, 100),
            "critic": torch.randn(8, 120),
        },
        batch_size=[8],
    )
    policy = ActorCriticFoothold(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        29,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
    )

    assert policy.act(obs).shape == (8, 29)
    assert policy.evaluate(obs).shape == (8, 1)
    assert policy.foothold_prediction.shape == (8, 6)


def test_touchdown_backfills_contiguous_airborne_segment() -> None:
    algorithm = object.__new__(PPOAMP)
    algorithm.device = "cpu"
    algorithm.foothold_cfg = {
        "supervision_group": "supervision",
        "target_group": "target",
        "valid_group": "valid",
        "step_dt": 0.02,
    }

    supervision = torch.zeros(4, 1, 10)
    supervision[..., 2] = 1.0
    supervision[:, 0, 0] = torch.tensor([0.0, 0.1, 0.2, 0.3])
    supervision[:, 0, 8] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    storage = type("Storage", (), {})()
    storage.observations = TensorDict(
        {
            "supervision": supervision,
            "target": torch.zeros(4, 1, 6),
            "valid": torch.zeros(4, 1, 2),
        },
        batch_size=[4, 1],
    )
    algorithm.storage = storage

    current_state = supervision[3].clone()
    current_state[:, 4] = 0.4
    current_state[:, 8] = 1.0
    algorithm._backfill_foothold_targets(
        step=3,
        previous_state=supervision[3],
        current_state=current_state,
        dones=torch.tensor([False]),
    )

    assert torch.allclose(storage.observations["target"][1:, 0, 0], torch.tensor([0.3, 0.2, 0.1]))
    assert torch.allclose(storage.observations["target"][1:, 0, 2], torch.tensor([0.06, 0.04, 0.02]))
    assert torch.equal(storage.observations["valid"][:, 0, 0], torch.tensor([0.0, 1.0, 1.0, 1.0]))
