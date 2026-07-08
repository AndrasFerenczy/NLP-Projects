import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
PAD_IDX = 0

def _use_amp(args) -> bool:
    # AMP in this repo is CUDA-only (uses torch.autocast + torch.cuda.amp.GradScaler).
    return bool(getattr(args, "amp", False) and DEVICE.type == "cuda")

def _amp_dtype_from_args(args) -> torch.dtype:
    dtype = getattr(args, "amp_dtype", "fp16")
    if dtype == "bf16":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("Warning: --amp_dtype bf16 requested but bf16 not supported; falling back to fp16.")
    return torch.float16

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=0,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    # Speed / iteration knobs
    parser.add_argument('--max_train_batches', type=int, default=-1,
                        help="If >0, cap number of train batches per epoch (for smoke tests).")
    parser.add_argument('--max_dev_batches', type=int, default=-1,
                        help="If >0, cap number of dev batches per eval (for smoke tests).")
    parser.add_argument('--no_dev_eval_generate', action='store_true',
                        help="If set, skip generation + metrics on dev (faster; uses dev loss for checkpointing).")
    parser.add_argument('--dev_eval_generate_start_epoch', type=int, default=0,
                        help="If >0, skip dev generation/metrics until this epoch (0 = always). "
                             "Ignored if --no_dev_eval_generate is set.")
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help="Max tokens to generate per example at eval/test time.")
    parser.add_argument('--num_beams', type=int, default=1,
                        help="Beam size for generation (1 = greedy).")
    parser.add_argument('--amp', action='store_true',
                        help="Enable CUDA automatic mixed precision (AMP) with GradScaler (CUDA only).")
    parser.add_argument('--amp_dtype', type=str, default="fp16", choices=["fp16", "bf16"],
                        help="AMP dtype (CUDA only). bf16 requires bf16-capable GPU.")
    parser.add_argument('--resume', type=str, default="none", choices=["none", "best", "last"],
                        help="Resume training from an existing checkpoint for this --experiment_name.")
    parser.add_argument('--canonicalize_sql_targets', action='store_true',
                        help="If set, canonicalize gold SQL targets (e.g., strip trailing ';', fix 'AND(' spacing).")
    parser.add_argument('--schema_conditioning', action='store_true',
                        help="If set, prepend a compact schema prefix to each NL input (filtered by token overlap).")
    parser.add_argument('--schema_max_tables', type=int, default=5,
                        help="Max tables to include in the schema prefix (only if --schema_conditioning).")
    parser.add_argument('--schema_max_columns_per_table', type=int, default=8,
                        help="Max columns per table in the schema prefix (only if --schema_conditioning).")
    parser.add_argument('--schema_include_utts', action='store_true',
                        help="If set, include column NL names (utts) in schema prefix (longer prompts).")
    parser.add_argument('--schema_separator', type=str, default=" | ",
                        help="Separator between schema prefix and NL question (only if --schema_conditioning).")
    parser.add_argument('--skip_test_inference', action='store_true',
                        help="If set, do not run test-set generation/record execution at the end (faster iteration).")

    args = parser.parse_args()
    args.dev_eval_generate = not args.no_dev_eval_generate
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    best_loss = float("inf")
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join('records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{args.experiment_name}_dev.pkl')
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        do_dev_generate = (
            args.dev_eval_generate and (epoch >= args.dev_eval_generate_start_epoch)
        )
        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args,
            model,
            dev_loader,
            gt_sql_path,
            model_sql_path,
            gt_record_path,
            model_record_path,
            dev_eval_generate=do_dev_generate,
        )
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        improved = False
        if do_dev_generate:
            if record_f1 > best_f1:
                best_f1 = record_f1
                improved = True
        else:
            if eval_loss < best_loss:
                best_loss = eval_loss
                improved = True

        if improved:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if improved:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()
    use_amp = _use_amp(args)
    amp_dtype = _amp_dtype_from_args(args)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for step, (encoder_input, encoder_mask, decoder_input, decoder_targets, _) in enumerate(tqdm(train_loader)):
        if args.max_train_batches > 0 and step >= args.max_train_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']

            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens
        
def eval_epoch(
    args,
    model,
    dev_loader,
    gt_sql_pth,
    model_sql_path,
    gt_record_path,
    model_record_path,
    dev_eval_generate=None,
):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    model.eval()
    criterion = nn.CrossEntropyLoss()
    use_amp = _use_amp(args)
    amp_dtype = _amp_dtype_from_args(args)

    total_loss = 0.0
    total_tokens = 0

    all_generated_sql = []

    with torch.inference_mode():
        for step, (encoder_input, encoder_mask, decoder_input, decoder_targets, initial_decoder_inputs) in enumerate(tqdm(dev_loader)):
            if args.max_dev_batches > 0 and step >= args.max_dev_batches:
                break
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            initial_decoder_inputs = initial_decoder_inputs.to(DEVICE)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    decoder_input_ids=decoder_input,
                )["logits"]

                non_pad = decoder_targets != PAD_IDX
                loss = criterion(logits[non_pad], decoder_targets[non_pad])
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = (total_loss / total_tokens) if total_tokens > 0 else 0.0
    if dev_eval_generate is None:
        dev_eval_generate = args.dev_eval_generate
    if not dev_eval_generate:
        return avg_loss, 0, 0, 0, 0

    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    with torch.inference_mode():
        for step, (encoder_input, encoder_mask, _, _, initial_decoder_inputs) in enumerate(tqdm(dev_loader)):
            if args.max_dev_batches > 0 and step >= args.max_dev_batches:
                break
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            initial_decoder_inputs = initial_decoder_inputs.to(DEVICE)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                generated_ids = model.generate(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    decoder_input_ids=initial_decoder_inputs,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=False,
                    use_cache=True,
                )
            generated_sql = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            generated_sql = [s.strip() for s in generated_sql]
            all_generated_sql.extend(generated_sql)

    save_queries_and_records(all_generated_sql, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(gt_sql_pth, model_sql_path, gt_record_path, model_record_path)
    error_rate = sum(1 for e in error_msgs if e) / max(1, len(error_msgs))

    return avg_loss, record_f1, record_em, sql_em, error_rate
        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    all_generated_sql = []
    use_amp = _use_amp(args)
    amp_dtype = _amp_dtype_from_args(args)

    with torch.inference_mode():
        for encoder_input, encoder_mask, initial_decoder_inputs in tqdm(test_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            initial_decoder_inputs = initial_decoder_inputs.to(DEVICE)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                generated_ids = model.generate(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    decoder_input_ids=initial_decoder_inputs,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=False,
                    use_cache=True,
                )
            generated_sql = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            generated_sql = [s.strip() for s in generated_sql]
            all_generated_sql.extend(generated_sql)

    save_queries_and_records(all_generated_sql, model_sql_path, model_record_path)

def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(
        args.batch_size,
        args.test_batch_size,
        canonicalize_sql_targets=args.canonicalize_sql_targets,
        schema_conditioning=args.schema_conditioning,
        schema_max_tables=args.schema_max_tables,
        schema_max_columns_per_table=args.schema_max_columns_per_table,
        schema_include_utts=args.schema_include_utts,
        schema_separator=args.schema_separator,
    )
    if args.resume == "best":
        model = load_model_from_checkpoint(args, best=True)
    elif args.resume == "last":
        model = load_model_from_checkpoint(args, best=False)
    else:
        model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args,
        model,
        dev_loader,
        gt_sql_path,
        model_sql_path,
        gt_record_path,
        model_record_path,
        dev_eval_generate=(not args.no_dev_eval_generate),
    )
    print(
        f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, "
        f"Record EM: {dev_record_em}, SQL EM: {dev_sql_em}"
    )
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    if not args.skip_test_inference:
        model_sql_path = os.path.join(f"results/t5_{model_type}_test.sql")
        model_record_path = os.path.join(f"records/t5_{model_type}_test.pkl")
        test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()
