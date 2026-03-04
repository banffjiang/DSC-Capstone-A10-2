import json
from dataclasses import dataclass
import math, torch
from transformers import  AutoTokenizer, AutoConfig, AutoModelForCausalLM, get_cosine_schedule_with_warmup #lr scheduler
from torch.utils.data import DataLoader
import re
import sys
from pathlib import Path

import argparse
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataloader_wiki import SimpleWikiPassageLoader
from data.dataloader import TextOnlyDataset
from data.build_tf_idf import tokens_set
from listener.listener.bertscore_listener import BERTScoreListener

from utils.utils import generate_summary, jaccard_ngrams, make_prompt, set_global_seed

RANDOM_SEED = 42 #for reproducibility
TOKEN_RE = re.compile(r"[a-z0-9]+")

def quick_generate_sample(
    policy,
    tokenizer,
    prompt,
    *,
    top_p,
    temperature,
    max_new_tokens,
    repetition_penalty,
    no_repeat_ngram_size,
):
    policy.eval()
    with torch.inference_mode():
        text = generate_summary(
            policy,
            tokenizer,
            prompt,
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            seed=RANDOM_SEED,
        )
    policy.train()
    return text

#grabs 3 random samples to evaluate model
def build_probe_prompts(dataset, *, n=3):
    prompts = []
    for i in range(min(n, len(dataset))):
        ex = dataset[i]
        passage = ex["passage"]
        prompts.append(make_prompt(passage))
    return prompts

def split_train_test_examples(examples, test_size, split_seed):
    if not 0.0 <= test_size < 1.0:
        raise ValueError(f"--test_size must be in [0.0, 1.0), got {test_size}")

    if len(examples) == 0:
        return [], []

    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(len(examples), generator=generator).tolist()
    shuffled = [examples[i] for i in indices]

    test_count = int(len(shuffled) * test_size)
    if test_size > 0.0:
        test_count = max(1, test_count)
    test_count = min(test_count, max(0, len(shuffled) - 1))

    train_examples = shuffled[test_count:]
    test_examples = shuffled[:test_count]
    return train_examples, test_examples

@dataclass
class PairBatch:
    #chosen
    ids_c: torch.Tensor
    attn_c: torch.Tensor
    labels_c: torch.Tensor
    #rejected
    ids_r: torch.Tensor
    attn_r: torch.Tensor
    labels_r: torch.Tensor

####################################################################################################################
# Stage 0: pretraining a GPT-2 randomly initialized model to get it to understand basic language syntax/context
# Just like how children learn words before they can speak, need to give model basic understanding before training 
####################################################################################################################

def collate_lm(tokenizer, texts, *, max_length):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    labels = input_ids.clone()
    labels[attn == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

def pre_training(
        output_dir,
        text_dataset=None,
        input = "data/train_100m_passages.jsonl",
        model_name="gpt2",
        block_size=256,
        batch_size=16,
        lr=1e-4,
        warmup_steps=500,
        total_steps=10000,
        probe_prompts=None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = AutoConfig.from_pretrained(model_name)
    config.pad_token_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_config(config).to(device)
    model.train()

    dl = DataLoader(
        TextOnlyDataset(text_dataset),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch_texts: collate_lm(tokenizer, batch_texts, max_length=block_size),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    step = 0
    for epoch in range(100): #will break when steps are reached
        for batch in dl:
            step += 1
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print({'step:': step, 'loss': float(loss)})

                if wandb.run is not None:
                    wandb.log(
                        {
                            "stage": "pretrain",
                            "pretrain/loss": float(loss.item()),
                            "pretrain/lr": float(lr_scheduler.get_last_lr()[0]),
                            "pretrain/step": step,
                            "pretrain/epoch": epoch,
                        },
                        step=step,
                    )

                samples = []
                for j, prompt in enumerate(probe_prompts):
                    gen = quick_generate_sample(
                        model,
                        tokenizer,
                        prompt,
                        top_p=0.9,
                        temperature=0.9,
                        max_new_tokens=64,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=3,
                    )
                    samples.append((j, gen))

                if step % 500 == 0:
                    print("\n=== Pre-training samples @ step", step, "===\n")
                    for j, gen in samples:
                        print(f"[probe {j}]\n{gen}\n")

            if step >= total_steps:
                # saves model and tokenizer will be loaded in the next stage
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                return step

####################################################################################################################
# Stage 1: training model to produce summaries that will make dpo work on being able to compare
####################################################################################################################

def tfidf_target(src: str, idf: dict, top_k: int = 8, min_len: int = 2):
    src = src.lower()
    toks = TOKEN_RE.findall(src)
    toks = [t for t in toks if len(t) >= min_len]
    if not toks:
        return "something"

    tf = {}
    for t in toks:
        tf[t] = tf.get(t, 0) + 1

    scores = {t: (1.0 + math.log(c)) * idf.get(t, 0.0) for t, c in tf.items()}
    keep = set([t for t, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]])

    out, seen = [], set()
    for t in toks:
        if t in keep and t not in seen:
            out.append(t); seen.add(t)
        if len(out) >= top_k:
            break
    return " ".join(out) if out else "something"

def collate_sft(batch, tokenizer, idf, block_size=256, top_k=8):
    prefix = "Keywords summary.\nText: "
    suffix = "\nOutput:"

    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]

    prompts, targets = [], []
    for e in batch:
        src = e["passage"]
        target = tfidf_target(src, idf, top_k=top_k)
        targets.append(target)

        target_ids = tokenizer(" " + target, add_special_tokens=False)["input_ids"]

        budget = block_size - (len(prefix_ids) + len(suffix_ids) + len(target_ids))
        if budget < 1:
            max_target = max(1, block_size - (len(prefix_ids) + len(suffix_ids) + 1))
            target_ids = target_ids[:max_target]
            budget = block_size - (len(prefix_ids) + len(suffix_ids) + len(target_ids))

        passage_ids = tokenizer(src, add_special_tokens=False)["input_ids"][:budget]
        prompt_ids = prefix_ids + passage_ids + suffix_ids
        prompt_text = tokenizer.decode(prompt_ids, clean_up_tokenization_spaces=False)
        prompts.append(prompt_text)

    full_texts = [p + " " + t for p, t in zip(prompts, targets)]

    toks = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=block_size,
    )
    input_ids = toks["input_ids"]
    attention_mask = toks["attention_mask"]

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100  # ignore padding

    # mask prompt tokens
    ptoks = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=block_size,
    )
    prompt_lens = ptoks["attention_mask"].sum(dim=1).tolist()

    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

#helper to print out the sample for tfidf target
def quick_tfidf_sample(dataset, idf, *, idx=0, top_k=8):
    ex = dataset[idx]
    src = ex["passage"]
    prompt = make_prompt(src)
    target = tfidf_target(src, idf, top_k=top_k)
    return src, prompt, target

def train_sft(
    step_offset,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    idf,
    optimizer,
    device,
    probe_prompts=None,
    *,
    block_size=256,
    top_k=8,
    batch_size=16,
    num_epochs=3,
):
    model.train()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_sft(b, tokenizer, idf, block_size=block_size, top_k=top_k),
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_sft(b, tokenizer, idf, block_size=block_size, top_k=top_k),
    )

    global_step = 0
    for epoch in range(num_epochs):
        for batch in train_loader:
            global_step += 1
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if global_step % 100 == 0:
                samples = []
                for j, prompt in enumerate(probe_prompts):
                    gen = quick_generate_sample(
                        model,
                        tokenizer,
                        prompt,
                        top_p=0.9,
                        temperature=0.9,
                        max_new_tokens=64,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=3,
                    )
                    samples.append((j, gen))

                print("\n=== [SFT] samples @ step", global_step, "===\n")
                for j, gen in samples:
                    print(f"[probe {j}]\n{gen}\n")
                
                lr = optimizer.param_groups[0]["lr"]
                print({"epoch": epoch, "step": global_step, "loss": float(loss), "lr": lr})
                if wandb.run is not None:
                    wandb.log(
                        {
                            "stage": "sft",
                            "sft/loss": float(loss),
                            "sft/lr": lr,
                            "sft/epoch": epoch,
                            "sft/step": global_step,
                        },
                        step=step_offset + global_step,
                    )
                
                if global_step % 500 == 0:
                    ex = train_dataset[0]
                    src = ex["passage"]
                    prompt = make_prompt(src)
                    target = tfidf_target(src, idf, top_k=top_k)

                    model_out = quick_generate_sample(
                        model,
                        tokenizer,
                        prompt,
                        top_p=0.8,                # tighter decoding for more stable diagnostics
                        temperature=0.7,
                        max_new_tokens=24,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=3,
                    )

                    print("\n=== [TFIDF DEBUG] step", global_step, "===\n")
                    print("SRC (first 200):", src[:200])
                    print("TFIDF TARGET:", target)
                    print("MODEL OUT:", model_out)
                    print()

            if global_step % 500 == 0:
                model.eval()
                total = 0.0
                nb = 0
                with torch.no_grad():
                    for ebatch in eval_loader:
                        ebatch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in ebatch.items()}
                        eloss = model(**ebatch).loss
                        total += eloss.detach().cpu().item()
                        nb += 1
                eval_loss = total / max(1, nb)
                model.train()

                print({"epoch": epoch, "step": global_step, "eval_loss": eval_loss})
                if wandb.run is not None:
                    wandb.log(
                        {"stage": "sft", "sft/eval_loss": eval_loss},
                        step=step_offset + global_step,
                    )


####################################################################################################################
# Stage 2: Training dpo with nll steps to replicate telegraphic speech
####################################################################################################################

####################################################################################################################

def parse_args():
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument("--idf_json", type=str, default="idf_wiki.json")
    parser.add_argument("--train_jsonl", type=str, default="data/train_100m_passages.jsonl")
    parser.add_argument("--test_size", type=float, default=0.1)

    # sft hyperparams
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--total_steps", type=int, default=5000)

    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    parser.add_argument("--skip_stage0", action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(RANDOM_SEED)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )

    with open(args.idf_json, "r") as f:
        payload = json.load(f)
    idf = payload["idf"]

    text = list(SimpleWikiPassageLoader(path=args.train_jsonl, limit=None))
    wiki_text = list(SimpleWikiPassageLoader(path="data/simple_wiki_passages.jsonl", limit=None))
    train_dataset, test_dataset = split_train_test_examples(wiki_text, test_size=args.test_size, split_seed=RANDOM_SEED)

    probe_prompts = build_probe_prompts(wiki_text, n=3)
    pretrain_steps = 0

    if not args.skip_stage0:
        pretrain_steps = pre_training(
            text_dataset=text,
            output_dir="/workspace/checkpoints/stage0_ckpt",
            input=args.train_jsonl,
            model_name="gpt2",
            block_size=args.block_size,
            batch_size=args.batch_size,
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            total_steps=20000,
            probe_prompts=probe_prompts,
            device=device,
        )

    tokenizer = AutoTokenizer.from_pretrained("/workspace/checkpoints/stage0_ckpt")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("/workspace/checkpoints/stage0_ckpt").to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ##testing to see that stage0_checkpoint makes
    prompt = "Hello, my name is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)

    print(tokenizer.decode(out[0]))

    train_sft(
        pretrain_steps,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        idf=idf,
        optimizer=optimizer,
        device=device,
        probe_prompts=probe_prompts,
        block_size=args.block_size,
        top_k=args.top_k,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
    )

    stage1 = "/workspace/checkpoints/stage1_sft_ckpt"
    model.save_pretrained(stage1)
    tokenizer.save_pretrained(stage1)

if __name__ == "__main__":
    main()
