"""Small CPU regression gates for hero export; full-checkpoint GPU parity is separate."""

from __future__ import annotations

import tempfile
import types
import unittest

import torch
from transformers import AutoModelForCausalLM

from olmo_core.nn.hf.config import _qk_norm_per_head_gains, _register_olmo3moe_auto_classes
from olmo_core.nn.hf.convert import convert_olmo3moe_state_from_hf, convert_olmo3moe_state_to_hf
from olmo_core.nn.layer_norm import RMSNorm
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM, Olmo3MoeRMSNorm


class HeroHFChecks(unittest.TestCase):
    """Exercise nontrivial per-head gains, GQA, legacy export, and cached attention."""

    def test_norm_matches_core(self):
        for dtype in (torch.float32, torch.bfloat16):
            for heads in (4, 8):
                torch.manual_seed(heads)
                x = torch.randn(2, 7, heads, 128).to(dtype)
                core = RMSNorm(size=128, weight_shape=(heads, 128), eps=1e-6).to(dtype)
                hf = Olmo3MoeRMSNorm(128, eps=1e-6, weight_shape=(heads, 128)).to(dtype)
                with torch.no_grad():
                    core.weight.copy_(torch.randn(heads, 128).to(dtype))
                    hf.weight.copy_(core.weight)
                torch.testing.assert_close(hf(x), core(x), rtol=0, atol=0)

    def test_gain_layout_validation(self):
        def attn(per_head):
            return types.SimpleNamespace(
                use_head_qk_norm=True,
                n_heads=8,
                n_kv_heads=4,
                head_dim=128,
                q_norm=types.SimpleNamespace(
                    weight=torch.ones((8, 128) if per_head else (128,)), bias=None
                ),
                k_norm=types.SimpleNamespace(
                    weight=torch.ones((4, 128) if per_head else (128,)), bias=None
                ),
            )

        self.assertTrue(_qk_norm_per_head_gains([attn(True), attn(True)]))
        self.assertFalse(_qk_norm_per_head_gains([attn(False)]))
        with self.assertRaises(NotImplementedError):
            _qk_norm_per_head_gains([attn(True), attn(False)])
        wrong = attn(True)
        wrong.k_norm.weight = torch.ones(8, 128)
        with self.assertRaises(NotImplementedError):
            _qk_norm_per_head_gains([wrong])
        with self.assertRaises(ValueError):
            Olmo3MoeConfig(qk_norm_per_head_gains=True, use_head_qk_norm=False)

    def test_tensor_roundtrip_reload_and_attention_cache(self):
        _register_olmo3moe_auto_classes()
        for per_head in (False, True):
            torch.manual_seed(42)
            cfg = Olmo3MoeConfig(
                vocab_size=64,
                hidden_size=32,
                attention_hidden_size=32,
                head_dim=8,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                dense_mlp_intermediate_size=24,
                moe_intermediate_size=12,
                shared_expert_intermediate_size=16,
                n_routed_experts=4,
                num_experts_per_tok=2,
                dense_layers_indices=[0],
                dense_layers_use_shared_expert=True,
                layer_types=["full_attention", "full_attention"],
                latent_moe_dim=16,
                use_head_qk_norm=True,
                qk_norm_per_head_gains=per_head,
                use_rope=False,
                scalable_softmax=True,
                use_peri_ln=True,
                max_position_embeddings=128,
                embed_norm=True,
            )
            cfg._attn_implementation = "eager"
            model = Olmo3MoeForCausalLM(cfg).eval()
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if "q_norm.weight" in name or "k_norm.weight" in name:
                        p.copy_(torch.randn_like(p) * 0.2 + 1)
            hf = model.state_dict()
            core = convert_olmo3moe_state_from_hf(cfg, hf)
            restored = convert_olmo3moe_state_to_hf(cfg, core)
            self.assertEqual(set(restored), set(hf))
            for name in hf:
                torch.testing.assert_close(restored[name], hf[name], rtol=0, atol=0)
            ids = torch.randint(0, 64, (2, 11))
            with torch.no_grad():
                full = model(ids, use_cache=False).logits
                first = model(ids[:, :7], use_cache=True)
                second = model(ids[:, 7:], past_key_values=first.past_key_values, use_cache=True)
            torch.testing.assert_close(full[:, 7:], second.logits, rtol=1e-5, atol=1e-6)
            with tempfile.TemporaryDirectory(prefix="hero-hf-unit-") as folder:
                model.save_pretrained(folder)
                reloaded = AutoModelForCausalLM.from_pretrained(
                    folder, trust_remote_code=True
                ).eval()
                self.assertEqual(reloaded.config.qk_norm_per_head_gains, per_head)
                with torch.no_grad():
                    torch.testing.assert_close(
                        reloaded(ids, use_cache=False).logits, full, rtol=1e-5, atol=1e-6
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
