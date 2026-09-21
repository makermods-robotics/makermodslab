"""Single-pass loading for complete MolmoAct2 LeRobot checkpoints.

The pinned LeRobot loader first loads the HF base, copies that base again, then
overwrites it with the LeRobot checkpoint. Build only its architecture here and
assign the final checkpoint once. This ships with the Lab's Modal source mount;
it does not patch installed packages or change other policy families.

Molmo validation below follows makermods-robotics/lerobot at
eaab69339120787948776e4354dcee09f501fd16 (Apache-2.0, Allen Institute for AI and
HuggingFace). Keep it aligned when updating the dependency pin.
"""

from copy import deepcopy
from pathlib import Path

from ._diagnostics import startup_stage


def _metadata_location(config):
    """Resolve processor/config assets without fetching base model weights."""
    local = Path(config.checkpoint_path).expanduser()
    if local.exists():
        return str(local)
    from huggingface_hub import snapshot_download

    from lerobot.policies.molmoact2.modeling_molmoact2 import _hf_token

    return snapshot_download(
        repo_id=config.checkpoint_path,
        revision=config.checkpoint_revision,
        force_download=bool(config.checkpoint_force_download),
        token=_hf_token(),
        allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja", "*.jinja2"],
    )


def _construct_molmo(policy_cls, config):
    import torch
    from transformers.initialization import no_init_weights

    from lerobot.policies.molmoact2.modeling_molmoact2 import (
        HFMolmoAct2Config,
        MolmoAct2ForConditionalGeneration,
        _torch_dtype,
        _validate_checkpoint_action_mode,
    )

    class ConfigOnlyMolmo(policy_cls):
        def _load_hf_model(self):
            hf_config = HFMolmoAct2Config.from_pretrained(self.config.checkpoint_path)
            # No storage, even for constructors that call torch.zeros before
            # registering a parameter. Persistent buffers come from the final
            # checkpoint; the rotary caches are empty and reconstructed below.
            with torch.device("meta"), no_init_weights():
                self.model = MolmoAct2ForConditionalGeneration._from_config(
                    hf_config, dtype=_torch_dtype(self.config.model_dtype)
                )

            max_dim = int(getattr(self.model.config, "max_action_dim", -1))
            if max_dim != int(self.config.expected_max_action_dim) or max_dim != 32:
                raise ValueError(
                    f"MolmoAct2 checkpoint max_action_dim mismatch: checkpoint={max_dim}, "
                    f"expected={self.config.expected_max_action_dim}; released models require 32."
                )
            if not hasattr(self.model.config, "max_action_horizon"):
                raise ValueError("MolmoAct2 HF checkpoints must define max_action_horizon.")
            self._override_loaded_max_action_horizon(int(self.config.chunk_size))
            if not hasattr(self.model.config, "action_mode"):
                raise ValueError("MolmoAct2 HF checkpoints must define action_mode.")
            _validate_checkpoint_action_mode(
                self.config,
                str(self.model.config.action_mode),
                has_action_expert=bool(getattr(self.model.config, "add_action_expert", False)),
            )
            if self.config.freeze_embedding:
                self._freeze_input_embeddings()
            if self.config.train_action_expert_only:
                self._freeze_non_action_expert_parameters()
            if self.config.gradient_checkpointing:
                self._enable_gradient_checkpointing()
            self.train(self.training)

    # The inherited constructor still applies norm-tag metadata, validates input
    # features/inference mode, and sets up queues/RTC. Its base path is now local
    # metadata, so that constructor cannot accidentally download the base weights.
    return ConfigOnlyMolmo(config)


def _assign_checkpoint(policy, filename, device):
    """Validate every key/shape first, then materialize each tensor only once."""
    import torch
    from safetensors import safe_open

    expected = policy.state_dict()
    with safe_open(str(filename), framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        missing, unexpected = set(expected) - keys, keys - set(expected)
        if missing or unexpected:
            raise ValueError(
                "Single-pass MolmoAct2 loading requires a complete LeRobot checkpoint. "
                f"Missing keys: {sorted(missing)[:8]}; unexpected keys: {sorted(unexpected)[:8]}"
            )
        for name, target in expected.items():
            shape = tuple(checkpoint.get_slice(name).get_shape())
            if shape != tuple(target.shape):
                raise ValueError(
                    f"MolmoAct2 checkpoint shape mismatch for {name}: "
                    f"checkpoint={shape}, expected={tuple(target.shape)}"
                )
        # On CUDA this converts each mapped CPU tensor directly to the final GPU
        # dtype. No full FP32 GPU copy or materialized CPU model is constructed.
        state = {
            name: checkpoint.get_tensor(name).to(device=device, dtype=target.dtype)
            for name, target in expected.items()
        }
    policy.load_state_dict(state, strict=True, assign=True)
    for module in policy.modules():
        for name in module._non_persistent_buffers_set:
            buffer = module._buffers[name]
            if buffer is not None and buffer.is_meta:
                if buffer.numel() != 0:
                    raise RuntimeError(f"Cannot reconstruct nonempty MolmoAct2 runtime buffer {name}")
                module.register_buffer(name, torch.empty_like(buffer, device=device), persistent=False)
    if any(t.is_meta for t in (*policy.parameters(), *policy.buffers())):
        raise RuntimeError("MolmoAct2 loading left unmaterialized tensors")
    policy.to(device)  # Also moves the small nonpersistent buffers.
    # assign=True replaces persistent buffers. Restore the rotary module's alias
    # used when resetting a dynamic RoPE cache, including on another device.
    for module in policy.modules():
        if hasattr(module, "original_inv_freq") and hasattr(module, "inv_freq"):
            module.original_inv_freq = module.inv_freq
    policy.eval()
    return policy


def load_pretrained_policy(policy_cls, policy_path, config, device, *, overridden=False):
    """Use the fast path for full Molmo checkpoints; preserve other loaders."""
    if config.type != "molmoact2" or getattr(config, "enable_lora_vlm", False):
        if config.type == "molmoact2":
            print("[policy] MolmoAct2 adapter config: using the existing pretrained loader", flush=True)
        return (
            policy_cls.from_pretrained(policy_path, config=config)
            if overridden
            else policy_cls.from_pretrained(policy_path)
        )

    from huggingface_hub import hf_hub_download

    from lerobot.policies.molmoact2.modeling_molmoact2 import _hf_token

    print("[policy] MolmoAct2 single-pass loader: skipping base weight loads", flush=True)
    config = deepcopy(config)
    with startup_stage("MolmoAct2 metadata (no base weights)"):
        config.checkpoint_path = _metadata_location(config)
    with startup_stage("MolmoAct2 architecture (empty parameters)"):
        policy = _construct_molmo(policy_cls, config)
        policy._mml_molmo_assets = config.checkpoint_path
    with startup_stage("MolmoAct2 final checkpoint resolution"):
        local = Path(policy_path)
        filename = (
            local / "model.safetensors"
            if local.is_dir()
            else hf_hub_download(repo_id=str(policy_path), filename="model.safetensors", token=_hf_token())
        )
    with startup_stage("MolmoAct2 final weights (single pass)"):
        return _assign_checkpoint(policy, filename, device)


def preprocessor_asset_overrides(policy, overrides):
    """Keep the saved pack processor from downloading the skipped base weights."""
    assets = getattr(policy, "_mml_molmo_assets", None)
    if assets is None:
        return overrides
    result = deepcopy(overrides)
    result.setdefault("molmoact2_pack_inputs", {}).update(
        checkpoint_path=assets, checkpoint_force_download=False
    )
    return result
