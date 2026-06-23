from src.utils.generation_utils import compute_generation_composite_score


def test_zero_forbidden_always_ranked_higher():
    # Even with lower valid_smiles_rate, forbidden==0 should dominate forbidden>0.
    score_zero_forbidden = compute_generation_composite_score(
        forbidden_domain_token_rate=0.0,
        valid_smiles_rate=0.50,
        val_loss=0.6,
    )
    score_nonzero_forbidden = compute_generation_composite_score(
        forbidden_domain_token_rate=0.01,
        valid_smiles_rate=0.99,
        val_loss=0.1,
    )
    assert score_zero_forbidden > score_nonzero_forbidden


def test_valid_smiles_breaks_tie_when_forbidden_zero():
    score_low_valid = compute_generation_composite_score(
        forbidden_domain_token_rate=0.0,
        valid_smiles_rate=0.60,
        val_loss=0.5,
    )
    score_high_valid = compute_generation_composite_score(
        forbidden_domain_token_rate=0.0,
        valid_smiles_rate=0.70,
        val_loss=0.5,
    )
    assert score_high_valid > score_low_valid


def test_val_loss_is_tie_break_when_forbidden_and_valid_same():
    score_high_loss = compute_generation_composite_score(
        forbidden_domain_token_rate=0.0,
        valid_smiles_rate=0.70,
        val_loss=1.0,
    )
    score_low_loss = compute_generation_composite_score(
        forbidden_domain_token_rate=0.0,
        valid_smiles_rate=0.70,
        val_loss=0.4,
    )
    assert score_low_loss > score_high_loss


def test_closure_reward_beats_same_validity():
    score_low_closure = compute_generation_composite_score(
        forbidden_domain_token_rate=0.0,
        valid_smiles_rate=0.80,
        mol_closure_rate=0.60,
        val_loss=0.4,
    )
    score_high_closure = compute_generation_composite_score(
        forbidden_domain_token_rate=0.0,
        valid_smiles_rate=0.80,
        mol_closure_rate=0.90,
        val_loss=0.4,
    )
    assert score_high_closure > score_low_closure


def test_degenerate_outputs_are_penalized():
    clean_score = compute_generation_composite_score(
        forbidden_domain_token_rate=0.0,
        valid_smiles_rate=0.80,
        mol_closure_rate=0.90,
        overlong_smiles_rate=0.0,
        repeat_like_rate=0.0,
        exclamation_degeneracy_rate=0.0,
        val_loss=0.4,
    )
    degenerate_score = compute_generation_composite_score(
        forbidden_domain_token_rate=0.0,
        valid_smiles_rate=0.80,
        mol_closure_rate=0.90,
        overlong_smiles_rate=0.10,
        repeat_like_rate=0.10,
        exclamation_degeneracy_rate=0.05,
        val_loss=0.4,
    )
    assert clean_score > degenerate_score


def test_missing_eos_is_penalized():
    clean_stop_score = compute_generation_composite_score(
        forbidden_domain_token_rate=0.0,
        valid_smiles_rate=0.80,
        mol_closure_rate=0.90,
        no_eos_rate=0.01,
        val_loss=0.4,
    )
    poor_stop_score = compute_generation_composite_score(
        forbidden_domain_token_rate=0.0,
        valid_smiles_rate=0.80,
        mol_closure_rate=0.90,
        no_eos_rate=0.20,
        val_loss=0.4,
    )
    assert clean_stop_score > poor_stop_score
