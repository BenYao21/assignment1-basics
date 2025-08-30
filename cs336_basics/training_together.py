import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.tokenizer import Tokenizer

def get_batch(data, batch_size, context_length, device):
    """
    Generates a batch of input and target sequences from the data.
    """
    n = len(data)
    ix = torch.randint(n - context_length, (batch_size,))
    inputs = torch.stack([torch.from_numpy(data[i : i + context_length].astype(np.int64)) for i in ix])
    targets = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + context_length].astype(np.int64)) for i in ix])
    return inputs.to(device), targets.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, context_length, device, eval_iters=10):
    """
    Estimates the loss on the training and validation sets.
    """
    model.eval()
    out = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, context_length, device)
            logits = model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model.")

    # Data arguments
    parser.add_argument('--train_data_path', type=str, default='data/owt_train.txt', help='Path to training data')
    parser.add_argument('--val_data_path', type=str, default='data/owt_valid.txt', help='Path to validation data')
    parser.add_argument('--tokenizer_path', type=str, default=None, help='Path to tokenizer model')

    # Model arguments
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1024, help='Dimension of feed-forward network')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--theta', type=float, default=10000.0, help='Theta for RoPE')

    # Optimizer arguments
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='AdamW beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='AdamW beta2')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--context_length', type=int, default=128, help='Context length')
    parser.add_argument('--num_iterations', type=int, default=10000, help='Number of training iterations')
    parser.add_argument('--eval_interval', type=int, default=250, help='Evaluate every N iterations')
    parser.add_argument('--log_interval', type=int, default=10, help='Log every N iterations')

    # Checkpointing arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')

    # Other arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--dtype', type=str, default='float16', help='Data type for training (float32, float16, bfloat16)')

    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(1337)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]

    # Data loading
    tokenizer = Tokenizer()
    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        tokenizer.load(args.tokenizer_path)
    else:
        # Fallback to character tokenizer if no model is provided
        with open(args.train_data_path, 'r') as f:
            text = f.read()
        tokenizer.train_char_tokenizer(text)
    
    def tokenize_file(path):
        with open(path, 'r') as f:
            text = f.read()
        return tokenizer.encode(text)

    train_ids = tokenize_file(args.train_data_path)
    val_ids = tokenize_file(args.val_data_path)
    
    train_data = np.memmap('train.mmap', dtype=np.uint16, mode='w+', shape=(len(train_ids),))
    train_data[:] = train_ids
    val_data = np.memmap('val.mmap', dtype=np.uint16, mode='w+', shape=(len(val_ids),))
    val_data[:] = val_ids
    
    print(f"Training data has {len(train_data):,} tokens.")
    print(f"Validation data has {len(val_data):,} tokens.")

    # Model and Optimizer
    model = TransformerLM(
        d_model=args.d_model,
        vocab_size=tokenizer.vocab_size(),
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        theta=args.theta,
        device=args.device,
        dtype=ptdtype
    ).to(args.device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    start_iter = 0
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'latest_checkpoint.pt')
        if os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            start_iter = load_checkpoint(checkpoint_path, model, optimizer)

    # Training loop
    model.train()
    start_time = time.time()
    for iteration in range(start_iter, args.num_iterations):
        # Evaluate loss periodically
        if iteration % args.eval_interval == 0 and iteration > 0:
            losses = estimate_loss(model, train_data, val_data, args.batch_size, args.context_length, args.device)
            print(f"Iter {iteration}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")

            # Save checkpoint
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_{iteration}.pt')
            latest_checkpoint_path = os.path.join(args.checkpoint_dir, 'latest_checkpoint.pt')
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            save_checkpoint(model, optimizer, iteration, latest_checkpoint_path)


        # Get a batch of data
        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, args.device)

        # Forward pass
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iteration % args.log_interval == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Iter {iteration}: Loss {loss.item():.4f}, Time per iter: {elapsed_time*1000/args.log_interval:.2f}ms")
            start_time = time.time()

if __name__ == '__main__':
    main()
