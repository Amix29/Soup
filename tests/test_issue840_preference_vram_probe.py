"""CPU contract tests for the preference-loss VRAM probe groundwork (#840)."""

from __future__ import annotations

from pathlib import Path


def test_dpo_batch_is_stretched_to_the_configured_length():
    import torch

    from soup_cli.trainer.stream_setup import _stretch_preference_probe_batch

    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [1, 4, 5]]),
        "attention_mask": torch.ones((2, 3), dtype=torch.long),
        "completion_mask": torch.tensor([[0, 1, 1], [0, 1, 1]]),
    }
    out = _stretch_preference_probe_batch(batch, seq_len=8)
    assert out["input_ids"].shape == (2, 8)
    assert out["attention_mask"].shape == (2, 8)
    assert out["completion_mask"].shape == (2, 8)
    assert out["completion_mask"][:, 3:].eq(1).all()


def test_orpo_and_simpo_full_sequences_are_stretched_but_prompt_is_not():
    import torch

    from soup_cli.trainer.stream_setup import _stretch_preference_probe_batch

    batch = {
        "chosen_input_ids": torch.tensor([[1, 2, 3, 4]]),
        "chosen_attention_mask": torch.ones((1, 4), dtype=torch.long),
        "chosen_labels": torch.tensor([[-100, -100, 3, 4]]),
        "rejected_input_ids": torch.tensor([[1, 2, 5]]),
        "rejected_attention_mask": torch.ones((1, 3), dtype=torch.long),
        "rejected_labels": torch.tensor([[-100, -100, 5]]),
        "prompt_input_ids": torch.tensor([[1, 2]]),
    }
    out = _stretch_preference_probe_batch(batch, seq_len=7)
    assert out["chosen_input_ids"].shape == (1, 7)
    assert out["rejected_input_ids"].shape == (1, 7)
    assert out["chosen_labels"][0, 4:].equal(out["chosen_input_ids"][0, 4:])
    assert out["rejected_labels"][0, 3:].equal(out["rejected_input_ids"][0, 3:])
    assert out["prompt_input_ids"].shape == (1, 2)


def test_kto_stretches_policy_and_kl_sequences():
    import torch

    from soup_cli.trainer.stream_setup import _stretch_preference_probe_batch

    batch = {}
    for prefix in ("", "KL_"):
        batch[prefix + "completion_input_ids"] = torch.tensor([[1, 2, 3], [1, 4, 5]])
        batch[prefix + "completion_attention_mask"] = torch.ones((2, 3), dtype=torch.long)
        batch[prefix + "completion_labels"] = torch.tensor([[-100, 2, 3], [-100, 4, 5]])

    batch["answer_input_ids"] = torch.tensor([[2, 3], [4, 5]])
    out = _stretch_preference_probe_batch(batch, seq_len=9)
    assert out["completion_input_ids"].shape == (2, 9)
    assert out["KL_completion_input_ids"].shape == (2, 9)
    assert out["completion_labels"][0, 3:].equal(out["completion_input_ids"][0, 3:])
    assert out["KL_completion_labels"][0, 3:].equal(out["KL_completion_input_ids"][0, 3:])
    assert out["answer_input_ids"].shape == (2, 2)


def test_all_four_preference_wrappers_call_the_post_trainer_probe():
    root = Path(__file__).resolve().parents[1]
    for name in ("dpo.py", "orpo.py", "simpo.py", "kto.py"):
        source = (root / "src" / "soup_cli" / "trainer" / name).read_text()
        assert "self._run_pending_stream_vram_probe()" in source, name


def test_pending_probe_executes_trainer_compute_loss(monkeypatch):
    import torch

    from soup_cli.trainer.stream_setup import StreamingSetupMixin, _ProbePlan
    from soup_cli.utils.layer_stream_runtime import StepPeak

    class Runtime:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    class Trainer:
        def __init__(self, model):
            self.model = model
            self.seen = None

        def get_train_dataloader(self):
            return [{
                "input_ids": torch.tensor([[1, 2, 3], [1, 4, 5]]),
                "attention_mask": torch.ones((2, 3), dtype=torch.long),
                "completion_mask": torch.tensor([[0, 1, 1], [0, 1, 1]]),
            }]

        def compute_loss(self, model, batch):
            self.seen = batch["input_ids"].shape
            return model.weight.sum()

    class Wrapper(StreamingSetupMixin):
        pass

    wrapper = Wrapper()
    wrapper.model = torch.nn.Linear(4, 4, bias=False)
    wrapper.trainer = Trainer(wrapper.model)
    wrapper.device = "cuda"
    wrapper._stream_runtime = Runtime()
    wrapper._pending_stream_vram_probe = _ProbePlan(
        task="dpo",
        batch_size=1,
        rows=2,
        seq_len=8,
        vocab_size=64,
        predicted_bytes=100,
        available_bytes=1_000,
    )

    def fake_measure(model, *, step, rows, seq_len, device):
        assert rows == 2
        assert seq_len == 8
        assert device == "cuda"
        loss = step()
        assert loss.requires_grad
        return StepPeak(
            peak_bytes=200,
            reserved_bytes=220,
            seconds=0.01,
            rows=rows,
            seq_len=seq_len,
        )

    monkeypatch.setattr(
        "soup_cli.utils.layer_stream_runtime.measure_loss_step_peak_bytes",
        fake_measure,
    )
    wrapper._run_pending_stream_vram_probe()
    assert wrapper.trainer.seen == torch.Size([2, 8])
    assert wrapper._pending_stream_vram_probe is None
    assert wrapper._stream_runtime.closed == 0
