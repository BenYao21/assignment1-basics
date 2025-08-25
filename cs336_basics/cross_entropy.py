import torch

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross entropy loss with numerical stability.

    Args:
        logits: A tensor of shape (*, vocab_size) containing the predicted logits.
        targets: A tensor of shape (*) containing the target indices.

    Returns:
        A scalar tensor containing the mean cross entropy loss.
    """
    # Subtract the largest element for numerical stability.
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    stable_logits = logits - max_logits

    # Compute log(sum(exp(logits))) in a numerically stable way.
    log_sum_exp_logits = torch.log(torch.exp(stable_logits).sum(dim=-1)) + max_logits.squeeze(-1)

    # Get the logits corresponding to the target indices.
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # The cross-entropy loss is log(sum(exp(logits))) - logits[target].
    loss = log_sum_exp_logits - target_logits

    # Return the average loss over all dimensions.
    return loss.mean()
