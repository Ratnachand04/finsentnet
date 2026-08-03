import torch

from finsentnet_pro.backend.models.text_branch import TextBranch


def test_text_branch_output_shapes():
    model = TextBranch(use_finbert=False)
    model.eval()

    batch_size, seq_len = 4, 48
    input_ids = torch.randint(1, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    sentence_vec, attn = model(input_ids, attention_mask)

    assert sentence_vec.shape == (batch_size, 512)
    assert attn.shape == (batch_size, 8, seq_len, seq_len)


def test_attention_mask_blocks_padded_keys():
    model = TextBranch(use_finbert=False)
    model.eval()

    input_ids = torch.randint(1, 30000, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    attention_mask[1, -4:] = 0
    input_ids[1, -4:] = 0

    _, attn = model(input_ids, attention_mask)

    # Keys that are padding should receive near-zero attention probability.
    masked_key_mass = attn[1, :, :, -4:].sum().item()
    unmasked_key_mass = attn[1, :, :, :-4].sum().item()

    assert masked_key_mass < 1e-4
    assert unmasked_key_mass > 0


def test_attention_head_dimensions_match_spec():
    model = TextBranch(use_finbert=False)

    assert model.self_attention.num_heads == 8
    assert model.self_attention.d_model == 512
    assert model.self_attention.d_k == 64
    assert model.self_attention.d_v == 64
