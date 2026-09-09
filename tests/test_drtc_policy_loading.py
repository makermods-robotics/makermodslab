"""Exercise the real Molmo loader with a tiny local model, no network or robot."""

from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from makermodslab.drtc._policy_loading import (
    _assign_checkpoint,
    _construct_molmo,
    _metadata_location,
    load_pretrained_policy,
    preprocessor_asset_overrides,
)


@pytest.fixture
def tiny_checkpoint(tmp_path):
    # LeRobot imports without policy extras, but leaves the HF model classes
    # as None. Skip only when the optional dependency itself is absent.
    pytest.importorskip("transformers")
    molmo = pytest.importorskip("lerobot.policies.molmoact2.modeling_molmoact2")
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

    hf_config = molmo.HFMolmoAct2Config(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "vocab_size": 64,
            "additional_vocab_size": 4,
            "max_position_embeddings": 32,
        },
        vit_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "image_default_input_size": [28, 28],
            "image_patch_size": 14,
            "image_num_pos": 5,
        },
        adapter_config={
            "vit_layers": [-1],
            "hidden_size": 32,
            "text_hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
        },
        action_expert_config={
            "hidden_size": 32,
            "num_layers": 1,
            "num_heads": 4,
            "ffn_multiple_of": 8,
            "timestep_embed_dim": 8,
        },
        max_action_horizon=4,
        max_action_dim=32,
        n_obs_steps=1,
        action_mode="continuous",
        flow_matching_num_steps=2,
    )
    base = tmp_path / "base"
    base.mkdir()
    model = molmo.MolmoAct2ForConditionalGeneration._from_config(hf_config, dtype=torch.float32)
    model.save_pretrained(base)
    config = MolmoAct2Config(
        checkpoint_path=str(base),
        device="cpu",
        model_dtype="float32",
        chunk_size=4,
        n_action_steps=4,
        action_mode="continuous",
        inference_action_mode="continuous",
        input_features={
            "observation.images.cam0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        enable_inference_cuda_graph=False,
    )
    policy = molmo.MolmoAct2Policy(deepcopy(config))
    # The final policy differs from the base: skipping the final load must fail
    # equality, even if a loader accidentally retained all base weights.
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(0.01)
    final = tmp_path / "final"
    final.mkdir()
    save_file({k: v.clone() for k, v in policy.state_dict().items()}, final / "model.safetensors")
    return molmo, config, final


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("rtc", [False, True])
def test_matches_legacy_weights_buffers_and_actions_without_reading_base(
    tiny_checkpoint, monkeypatch, dtype, rtc
):
    molmo, config, final = tiny_checkpoint
    config.model_dtype = dtype
    if rtc:
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        config.rtc_config = RTCConfig(execution_horizon=2)
    legacy = molmo.MolmoAct2Policy.from_pretrained(final, config=deepcopy(config), strict=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("The fast loader must not read base weights")

    monkeypatch.setattr(molmo.MolmoAct2ForConditionalGeneration, "from_pretrained", forbidden)
    monkeypatch.setattr(molmo, "_strict_load_safetensors_weights", forbidden)
    import safetensors

    open_checkpoint = safetensors.safe_open
    reads = []

    @contextmanager
    def tracked_open(*args, **kwargs):
        with open_checkpoint(*args, **kwargs) as source:

            def get_tensor(name):
                reads.append(name)
                return source.get_tensor(name)

            yield SimpleNamespace(keys=source.keys, get_slice=source.get_slice, get_tensor=get_tensor)

    monkeypatch.setattr(safetensors, "safe_open", tracked_open)
    fast = load_pretrained_policy(molmo.MolmoAct2Policy, final, config, "cpu")
    assert len(reads) == len(set(reads)) == len(legacy.state_dict())
    assert fast.training is False
    assert fast.config is not config
    assert fast.state_dict().keys() == legacy.state_dict().keys()
    for name, expected in legacy.state_dict().items():
        actual = fast.state_dict()[name]
        assert actual.dtype == expected.dtype, name
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)
    assert not any(t.is_meta for t in (*fast.parameters(), *fast.buffers()))
    for module in fast.modules():
        if hasattr(module, "original_inv_freq"):
            assert module.original_inv_freq is module.inv_freq

    batch = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    actions = []
    for policy in (legacy, fast):
        kwargs = (
            {"inference_delay": 1, "execution_horizon": 2, "prev_chunk_left_over": torch.zeros(1, 4, 6)}
            if rtc
            else {}
        )
        actions.append(
            policy.predict_action_chunk(batch, generator=torch.Generator().manual_seed(37), **kwargs)
        )
    torch.testing.assert_close(actions[0], actions[1], rtol=0, atol=0)


@pytest.mark.parametrize("defect", ["missing", "unexpected", "shape"])
def test_refuses_incomplete_or_incompatible_checkpoint(tiny_checkpoint, defect):
    molmo, config, final = tiny_checkpoint
    policy = _construct_molmo(molmo.MolmoAct2Policy, config)
    from safetensors.torch import load_file

    state = load_file(final / "model.safetensors")
    name = next(iter(state))
    if defect == "missing":
        state.pop(name)
    elif defect == "unexpected":
        state["extra.weight"] = torch.zeros(1)
    else:
        state[name] = torch.zeros(1)
    save_file(state, final / "broken.safetensors")
    with pytest.raises(ValueError, match="complete LeRobot checkpoint|shape mismatch"):
        _assign_checkpoint(policy, final / "broken.safetensors", "cpu")
    assert all(p.is_meta for p in policy.parameters())  # Refused before any assignment.


def test_metadata_download_cannot_fetch_weight_shards(monkeypatch):
    pytest.importorskip("lerobot.policies.molmoact2.modeling_molmoact2")
    import fnmatch

    download = Mock(return_value="/cached/assets")
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    config = SimpleNamespace(
        checkpoint_path="example/molmo", checkpoint_revision="abc", checkpoint_force_download=False
    )
    assert _metadata_location(config) == "/cached/assets"
    patterns = download.call_args.kwargs["allow_patterns"]
    for name in ["model.safetensors", "model-00001-of-00006.safetensors", "pytorch_model.bin"]:
        assert not any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
    assert download.call_args.kwargs["revision"] == "abc"


@pytest.mark.parametrize("kind,lora", [("pi0", False), ("molmoact2", True)])
@pytest.mark.parametrize("overridden", [False, True])
def test_other_policies_and_adapters_retain_existing_loader(kind, lora, overridden):
    cls = SimpleNamespace(from_pretrained=Mock(return_value="loaded"))
    cfg = SimpleNamespace(type=kind, enable_lora_vlm=lora)
    assert load_pretrained_policy(cls, "checkpoint", cfg, "cpu", overridden=overridden) == "loaded"
    cls.from_pretrained.assert_called_once_with("checkpoint", **({"config": cfg} if overridden else {}))


def test_preprocessor_reuses_metadata_without_losing_extra_camera_roles():
    original = {"molmoact2_pack_inputs": {"image_keys": ["cam0", "cam1", "cam2"]}}
    updated = preprocessor_asset_overrides(SimpleNamespace(_mml_molmo_assets="/assets"), original)
    assert updated["molmoact2_pack_inputs"] == {
        "image_keys": ["cam0", "cam1", "cam2"],
        "checkpoint_path": "/assets",
        "checkpoint_force_download": False,
    }
    assert "checkpoint_path" not in original["molmoact2_pack_inputs"]
    assert preprocessor_asset_overrides(SimpleNamespace(), original) is original


@pytest.mark.parametrize("entrypoint", ["policy", "policy_rtc"])
def test_entrypoints_use_single_pass_and_preserve_operator_overrides(
    tiny_checkpoint, monkeypatch, entrypoint
):
    import importlib

    pytest.importorskip("livekit.portal")
    molmo, config, final = tiny_checkpoint
    config._save_pretrained(final)
    module = importlib.import_module(f"makermodslab.drtc.{entrypoint}")
    processors = Mock(return_value=("pre", "post"))
    monkeypatch.setattr(module, "make_pre_post_processors", processors)
    monkeypatch.setattr(
        molmo.MolmoAct2Policy, "from_pretrained", Mock(side_effect=AssertionError("legacy path"))
    )
    policy, pre, post = module.load_policy(
        str(final),
        torch.device("cpu"),
        task="pick up the block",
        model_dtype="bfloat16",
        flow_steps=3,
        extra_image_roles=["cam1"],
    )
    assert (pre, post) == ("pre", "post")
    assert policy.config.num_inference_steps == 3
    assert all(p.dtype == torch.bfloat16 for p in policy.parameters())
    pack = processors.call_args.kwargs["preprocessor_overrides"]["molmoact2_pack_inputs"]
    assert pack["checkpoint_path"] == config.checkpoint_path
    assert pack["image_keys"] == ["observation.images.cam0", "observation.images.cam1"]
