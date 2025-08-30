import torch
from .embedding import Embedding
#data: int list
#output: int list to tensor, tensor of corresponding next tokens
def data_loading(data: list[int], batch_size: int, context_length: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of input and target sequences from the tokenized data.

    Args:
        data: A list of integers representing the tokenized data.
        batch_size: The number of sequences in a batch.
        context_length: The length of each sequence.
        device: The device to place the output tensors on (e.g., 'cpu' or 'cuda').

    Returns:
        A tuple of two tensors:
        - inputs: A tensor of shape (batch_size, context_length) with input sequences.
        - targets: A tensor of shape (batch_size, context_length) with target sequences,
                   which are the next tokens corresponding to the inputs.
    """
    data_tensor = torch.tensor(data, dtype=torch.long)
    n = len(data_tensor)

    # Randomly select batch_size starting indices
    # The maximum starting index must ensure that there are `context_length` tokens for the target sequence.
    # The last index accessed will be `i + context_length`, which must be less than `n`.
    # So, `i` must be less than `n - context_length`.
    ix = torch.randint(n - context_length, (batch_size,))

    # Create input sequences by taking `context_length` tokens from each starting index.
    # Using torch.stack is a clean way to build the batch.
    inputs = torch.stack([data_tensor[i : i + context_length] for i in ix])

    # Create target sequences, which are shifted by one token to the right.
    targets = torch.stack([data_tensor[i + 1 : i + 1 + context_length] for i in ix])

    # Move tensors to the specified device
    inputs, targets = inputs.to(device), targets.to(device)

    return inputs, targets