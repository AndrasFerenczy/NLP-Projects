import os, argparse, random, json, math, heapq, re
from collections import Counter, defaultdict
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM
from transformers import BitsAndBytesConfig

from utils import set_random_seeds, compute_metrics, save_queries_and_records
from prompting_utils import read_schema, extract_sql_query, save_logs
from load_data import load_prompting_data

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


_DOMAIN_KEYWORDS = {
    "flight",
    "airline",
    "airport",
    "city",
    "state",
    "fare",
    "cost",
    "price",
    "arrival",
    "depart",
    "departure",
    "stop",
    "stops",
    "connections",
    "meal",
    "service",
    "time",
    "day",
    "month",
    "year",
    "discount",
    "restriction",
    "aircraft",
    "capacity",
}


def _tokenize(text: str) -> list[str]:
    # Simple, dependency-free tokenizer for selection heuristics.
    # Keeps alphanumerics and underscores to match schema/table tokens.
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


class _TfidfIndex:
    def __init__(self, documents: list[str]):
        self._n_docs = len(documents)
        self._idf: dict[str, float] = {}
        self._postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._doc_norms: list[float] = [0.0] * self._n_docs

        tokenized_docs = [_tokenize(d) for d in documents]
        df: Counter[str] = Counter()
        for toks in tokenized_docs:
            df.update(set(toks))

        self._idf = {t: math.log((self._n_docs + 1) / (c + 1)) + 1.0 for t, c in df.items()}

        # Build normalized TF-IDF vectors and an inverted index (token -> (doc, weight)).
        for doc_idx, toks in enumerate(tokenized_docs):
            tf = Counter(toks)
            weights: dict[str, float] = {}
            for t, c in tf.items():
                idf = self._idf.get(t)
                if idf is None:
                    continue
                weights[t] = (1.0 + math.log(c)) * idf
            norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
            self._doc_norms[doc_idx] = norm
            for t, w in weights.items():
                self._postings[t].append((doc_idx, w / norm))

    def topk(self, query: str, k: int) -> list[int]:
        if k <= 0:
            return []
        q_toks = _tokenize(query)
        if not q_toks:
            return []
        tf = Counter(q_toks)
        q_weights: dict[str, float] = {}
        for t, c in tf.items():
            idf = self._idf.get(t)
            if idf is None:
                continue
            q_weights[t] = (1.0 + math.log(c)) * idf
        q_norm = math.sqrt(sum(w * w for w in q_weights.values())) or 1.0
        q_weights = {t: w / q_norm for t, w in q_weights.items()}

        scores: dict[int, float] = defaultdict(float)
        for t, q_w in q_weights.items():
            for doc_idx, d_w in self._postings.get(t, []):
                scores[doc_idx] += q_w * d_w

        if not scores:
            return []
        return [idx for idx, _ in heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])]


def get_args():
    '''
    Arguments for prompting. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(
        description='Text-to-SQL experiments with prompting.')

    parser.add_argument('-s', '--shot', type=int, default=0,
                        help='Number of examples for k-shot learning (0 for zero-shot)')
    parser.add_argument('-m', '--model', type=str, default='gemma-1b',
                        help='Model to use for prompting')
    parser.add_argument('-q', '--quantization', action='store_true',
                        help='Use a quantized version of the model (e.g. 4bits)')

    parser.add_argument('--eval_split', type=str, default='dev', choices=['dev', 'test', 'both'],
                        help='Which split to run prompting on')
    parser.add_argument('--selection', type=str, default='random', choices=['random', 'tfidf', 'heuristic'],
                        help='Few-shot example selection strategy')

    parser.add_argument('--schema_path', type=str, default='data/flight_database.schema',
                        help='Path to schema file for grounding')
    parser.add_argument('--schema_mode', type=str, default='compact', choices=['none', 'compact', 'full'],
                        help='How much schema to include in the prompt')
    parser.add_argument('--schema_max_chars', type=int, default=4000,
                        help='Maximum characters of schema text to include')

    parser.add_argument('--max_new_tokens', type=int, default=192,
                        help='Maximum new tokens to generate per example')
    parser.add_argument('--do_sample', action='store_true',
                        help='Use sampling instead of greedy decoding')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='Sampling temperature (only if --do_sample)')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Nucleus sampling p (only if --do_sample)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed to help reproducibility')
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")
    parser.add_argument('--output_prefix', type=str, default='gemma',
                        help='Output prefix for results/{prefix}_{split}.sql and records/{prefix}_{split}.pkl')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for JSONL logs (optional)')
    parser.add_argument('--save_jsonl', action='store_true',
                        help='Save per-example JSONL logs (prompt/output/extraction/errors)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Optional limit on number of examples (debugging)')
    args = parser.parse_args()
    return args


def _select_icl_examples(
    sentence: str,
    *,
    k: int,
    strategy: str,
    rng: random.Random,
    train_x: list[str],
    train_y: list[str],
    tfidf_index: _TfidfIndex | None,
    train_token_sets: list[set[str]] | None,
) -> list[tuple[str, str]]:
    if k <= 0:
        return []
    if k > len(train_x):
        raise ValueError(f"k={k} is larger than training set size ({len(train_x)})")

    if strategy == "random":
        idxs = rng.sample(range(len(train_x)), k)
        return [(train_x[i], train_y[i]) for i in idxs]

    if strategy == "tfidf":
        if tfidf_index is None:
            raise ValueError("tfidf strategy requires a tfidf_index")
        idxs = tfidf_index.topk(sentence, k)
        if len(idxs) < k:
            # Rare edge case (no overlap): pad with random samples not already chosen.
            chosen = set(idxs)
            for i in rng.sample(range(len(train_x)), k):
                if i not in chosen:
                    idxs.append(i)
                    chosen.add(i)
                if len(idxs) >= k:
                    break
        return [(train_x[i], train_y[i]) for i in idxs[:k]]

    if strategy == "heuristic":
        if train_token_sets is None:
            raise ValueError("heuristic strategy requires train_token_sets")
        sent_tokens = set(_tokenize(sentence))
        domain_tokens = sent_tokens & _DOMAIN_KEYWORDS

        scored: list[tuple[int, int]] = []
        for i, cand_tokens in enumerate(train_token_sets):
            overlap = len(sent_tokens & cand_tokens)
            domain_overlap = len(domain_tokens & cand_tokens)
            score = overlap + 2 * domain_overlap
            scored.append((score, i))
        # Break ties deterministically by index.
        scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
        idxs = [i for _, i in scored[:k]]
        return [(train_x[i], train_y[i]) for i in idxs]

    raise ValueError(f"Unknown selection strategy: {strategy}")


def create_prompt(sentence: str, *, schema_text: str, examples: list[tuple[str, str]] | None = None) -> str:
    '''
    Function for creating a prompt for zero or few-shot prompting.

    Add/modify the arguments as needed.

    Inputs:
        * sentence (str): A text string
        * k (int): Number of examples in k-shot prompting
    '''
    examples = examples or []

    parts: list[str] = []
    parts.append(
        """You will write ONE SQLite query that answers the question about
        the flight database.
        Hard constraints (must follow):
        1) Output ONLY the SQL query (no markdown, no explanation, no comments).
        2) Use only tables and columns that appear in the schema.
        3) Include join predicates for every join you introduce.
        4) Respect all constraints in the question (city, airline, date/day, fare rules, one-
        way vs round-trip, etc.).
        5) If the question asks for "most/least/cheapest/expensive", use MIN/MAX (or ORDER
        BY ... LIMIT 1) appropriately.
        
        
        SCHEMA (authoritative):
        {schema}

        FEW-SHOT EXAMPLES (if any):
        {examples}

        Now answer:

        NL: {nl}
        SQL:"""
        )

    if schema_text:
        parts.append("\nSCHEMA:\n" + schema_text.strip())

    if examples:
        ex_lines = ["\nEXAMPLES:"]
        for nl, sql in examples:
            ex_lines.append(f"NL: {nl}")
            ex_lines.append(f"SQL: {sql}")
            ex_lines.append("")  # separator
        parts.append("\n".join(ex_lines).rstrip())

    parts.append(f"\nNL: {sentence}\nSQL:")
    return "\n".join(parts).strip()

def _format_examples(examples: list[tuple[str, str]]) -> str:
    if not examples:
        return ""
    ex_lines = ["EXAMPLES:"]
    for nl, sql in examples:
        ex_lines.append(f"NL: {nl}")
        ex_lines.append(f"SQL: {sql}")
        ex_lines.append("")  # separator
    return "\n".join(ex_lines).rstrip()


def create_prompt_with_template(
    sentence: str,
    *,
    schema_text: str,
    examples: list[tuple[str, str]] | None = None,
    prompt_template: str,
) -> str:
    """
    Render a prompt from a user-provided template string.

    Available placeholders:
      - {schema}: schema text (may be empty)
      - {examples}: formatted few-shot examples (may be empty)
      - {nl}: the natural language question

    This is intended for interactive prompt iteration (e.g., in Colab) without editing code.
    """
    examples = examples or []
    return prompt_template.format(
        schema=(schema_text or "").strip(),
        examples=_format_examples(examples),
        nl=sentence,
    ).strip()


def exp_kshot(
    tokenizer,
    model,
    inputs: list[str],
    *,
    k: int,
    selection: str,
    seed: int,
    schema_text: str,
    train_x: list[str],
    train_y: list[str],
    tfidf_index: _TfidfIndex | None,
    train_token_sets: list[set[str]] | None,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    jsonl_log_path: str | None,
    system_prompt: str | None = None,
    prompt_template: str | None = None,
):
    '''
    k-shot prompting experiments using the provided model and tokenizer. 
    This function generates SQL queries from text prompts and evaluates their accuracy.

    Add/modify the arguments and code as needed.

    Inputs:
        * tokenizer
        * model
        * inputs (List[str]): A list of text strings
        * k (int): Number of examples in k-shot prompting
    '''
    rng = random.Random(seed)

    raw_outputs = []
    extracted_queries = []

    log_f = None
    if jsonl_log_path:
        os.makedirs(os.path.dirname(jsonl_log_path), exist_ok=True)
        log_f = open(jsonl_log_path, "w")

    for i, sentence in tqdm(list(enumerate(inputs)), total=len(inputs)):
        # Make selection stable per-example while still seed-controlled.
        per_ex_rng = random.Random((seed * 1_000_003) ^ i)
        examples = _select_icl_examples(
            sentence,
            k=k,
            strategy=selection,
            rng=per_ex_rng,
            train_x=train_x,
            train_y=train_y,
            tfidf_index=tfidf_index,
            train_token_sets=train_token_sets,
        )
        if prompt_template:
            prompt = create_prompt_with_template(
                sentence,
                schema_text=schema_text,
                examples=examples,
                prompt_template=prompt_template,
            )
        else:
            prompt = create_prompt(sentence, schema_text=schema_text, examples=examples)

        if system_prompt is None:
            system_prompt = (
                "You are a meticulous SQLite query writer for a flight database. "
                "You must return exactly ONE executable SQLite query and nothing else."
            )

        messages=[{
            "role": "system",
            "content": system_prompt, # you may want to prompt engineer this
        },
        {
            "role": "user",
            "content": prompt,
        }
        ]
        input_tokenized = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        input_len = int(input_tokenized["input_ids"].shape[-1])
        with torch.inference_mode():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            outputs = model.generate(**input_tokenized, **gen_kwargs)
        gen_ids = outputs[0][input_len:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)
        raw_outputs.append(response)

        # Extract the SQL query
        extracted_query = extract_sql_query(response)
        extracted_queries.append(extracted_query)

        if log_f is not None:
            log_obj = {
                "idx": i,
                "nl": sentence,
                "k": k,
                "selection": selection,
                "examples": [{"nl": ex_nl, "sql": ex_sql} for ex_nl, ex_sql in examples],
                "prompt": prompt,
                "raw_response": response,
                "extracted_sql": extracted_query,
            }
            log_f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")
 

    if log_f is not None:
        log_f.close()
    return raw_outputs, extracted_queries


def eval_outputs(*, gt_sql_path: str, model_sql_path: str, gt_record_path: str, model_record_path: str):
    '''
    Evaluate the outputs of the model by computing the metrics.

    Add/modify the arguments and code as needed.
    '''
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_path,
        model_sql_path,
        gt_record_path,
        model_record_path,
    )

    nonempty = sum(1 for m in model_error_msgs if m)
    error_rate = nonempty / max(1, len(model_error_msgs))
    return sql_em, record_em, record_f1, model_error_msgs, error_rate


def initialize_model_and_tokenizer(model_name, to_quantize=False):
    '''
    Args:
        * model_name (str): Model name (e.g., "gemma-1b").
        * to_quantize (bool): Use a quantized version of the model (e.g. 4bits)
    
    To access to the model on HuggingFace, you need to log in and review the 
    conditions and access the model's content.
    '''
    # If the model is gated on Hugging Face, you must:
    #  1) Visit the model card (e.g., https://huggingface.co/google/gemma-3-1b-it) and request/accept access.
    #  2) Provide a token from an authorized HF account (e.g., via `huggingface_hub.login(...)` on Colab).
    hf_token = os.environ.get("HF_TOKEN") or None
    if model_name == "gemma-1b":
        model_id = "google/gemma-3-1b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Native weights exported in bfloat16 precision, but choose a safe dtype per device.
        if DEVICE.type == "cuda":
            dtype = torch.bfloat16
        elif DEVICE.type == "mps":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        if to_quantize:
            if DEVICE.type != "cuda":
                raise ValueError("4-bit quantization requires CUDA")
            try:
                import bitsandbytes  # noqa: F401
            except Exception as e:
                raise ImportError(
                    "bitsandbytes is required for --quantization. Install it in your environment/Colab."
                ) from e
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4", # 4-bit quantization
                bnb_4bit_compute_dtype=dtype,
            )
            model = Gemma3ForCausalLM.from_pretrained(
                model_id,
                quantization_config=nf4_config,
                device_map="auto",
                token=hf_token,
            ).eval()
        else:
            model = (
                Gemma3ForCausalLM.from_pretrained(model_id, torch_dtype=dtype, token=hf_token)
                .to(DEVICE)
                .eval()
            )
    elif model_name == "gemma-27b":
        raise NotImplementedError(
            "gemma-27b is not wired up in this starter; use --model gemma-1b (or extend initialize_model_and_tokenizer)."
        )
    else:
        raise NotImplementedError(f"Model {model_name} is not implemented in this template.")
        # #you can extend this to use 4B and 12B versions. 


    return tokenizer, model


def main():
    '''
    Note: this code serves as a basic template for the prompting task. You can but 
    are not required to use this pipeline.
    You can design your own pipeline, and you can also modify the code below.
    '''
    args = get_args()
    shot = args.shot
    model_name = args.model
    to_quantize = args.quantization
    experiment_name = args.experiment_name

    set_random_seeds(args.seed)

    data_folder = 'data'
    train_x, train_y, dev_x, dev_y, test_x = load_prompting_data(data_folder)

    if args.limit is not None:
        if args.eval_split in {"dev", "both"}:
            dev_x, dev_y = dev_x[: args.limit], dev_y[: args.limit]
        if args.eval_split in {"test", "both"}:
            test_x = test_x[: args.limit]

    schema_text = read_schema(args.schema_path, mode=args.schema_mode, max_chars=args.schema_max_chars)

    tfidf_index = None
    train_token_sets = None
    if args.shot > 0:
        if args.selection == "tfidf":
            tfidf_index = _TfidfIndex(train_x)
        if args.selection == "heuristic":
            train_token_sets = [set(_tokenize(x)) for x in train_x]

    # Model and tokenizer
    tokenizer, model = initialize_model_and_tokenizer(model_name, to_quantize)

    splits: list[str]
    if args.eval_split == "both":
        splits = ["dev", "test"]
    else:
        splits = [args.eval_split]

    for split in splits:
        eval_x = dev_x if split == "dev" else test_x

        jsonl_log_path = None
        if args.save_jsonl:
            jsonl_log_path = os.path.join(
                args.log_dir, f"{args.output_prefix}_{experiment_name}_{split}.jsonl"
            )

        _, extracted_queries = exp_kshot(
            tokenizer,
            model,
            eval_x,
            k=shot,
            selection=args.selection,
            seed=args.seed,
            schema_text=schema_text,
            train_x=train_x,
            train_y=train_y,
            tfidf_index=tfidf_index,
            train_token_sets=train_token_sets,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            jsonl_log_path=jsonl_log_path,
        )

        model_sql_path = os.path.join("results", f"{args.output_prefix}_{split}.sql")
        model_record_path = os.path.join("records", f"{args.output_prefix}_{split}.pkl")
        save_queries_and_records(extracted_queries, model_sql_path, model_record_path)

        if split == "dev":
            gt_sql_path = os.path.join("data", "dev.sql")
            gt_record_path = os.path.join("records", "ground_truth_dev.pkl")
            sql_em, record_em, record_f1, model_error_msgs, error_rate = eval_outputs(
                gt_sql_path=gt_sql_path,
                model_sql_path=model_sql_path,
                gt_record_path=gt_record_path,
                model_record_path=model_record_path,
            )
            print("dev set results: ")
            print(f"Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
            print(f"dev set SQL error rate: {error_rate*100:.2f}%")

            if experiment_name:
                os.makedirs(args.log_dir, exist_ok=True)
                log_path = os.path.join(args.log_dir, f"{args.output_prefix}_{experiment_name}_dev.txt")
                save_logs(log_path, sql_em, record_em, record_f1, model_error_msgs)
        else:
            # No ground truth for test: still report the execution error rate.
            import pickle
            with open(model_record_path, "rb") as f:
                _, error_msgs = pickle.load(f)
            error_rate = sum(1 for m in error_msgs if m) / max(1, len(error_msgs))
            print(f"test set SQL error rate (no GT metrics): {error_rate*100:.2f}%")


if __name__ == "__main__":
    main()
