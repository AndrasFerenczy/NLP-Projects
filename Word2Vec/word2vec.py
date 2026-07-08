import argparse
import os
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.corpus import wordnet
from tqdm import tqdm

from load_data import load_data_word2vec
from utils import compute_spearman_correlation, get_similarity_scores

# Determine device for training
ACTIVE_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Word2vec(nn.Module):
    """
    Word2Vec Skip-Gram Model Structure with Negative Sampling
    Includes subword aggregation if dimensions permit.
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_ctx_tokens: int = None):
        super(Word2vec, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_vocab_size = num_ctx_tokens if num_ctx_tokens is not None else vocab_size

        # Core embedding layers
        self.target_matrix = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.context_matrix = nn.Embedding(self.context_vocab_size, self.embed_dim, padding_idx=0)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights using a uniform distribution relative to embed_dim."""
        init_range = 0.5 / (self.embed_dim ** 0.5)
        with torch.no_grad():
            self.target_matrix.weight.uniform_(-init_range, init_range)
            # Ensure padding remains zeroed
            self.target_matrix.weight[0].fill_(0.0)
            # Context weights start at zero
            self.context_matrix.weight.fill_(0.0)

    def forward(self, input_tokens: torch.Tensor, output_tokens: torch.Tensor,
                negative_context: Optional[torch.Tensor] = None):
        """
        Performs the forward pass yielding target, context, and negative context embeddings.
        """
        # (BatchSize, EmbedDim) or (BatchSize, SubwordCount, EmbedDim)
        tgt_emb = self.target_matrix(input_tokens)
        
        # Aggregate subwords by sum if they exist
        if tgt_emb.dim() == 3:
            tgt_emb = torch.sum(tgt_emb, dim=1)

        ctx_emb = self.context_matrix(output_tokens)  # (BatchSize, EmbedDim)

        neg_emb = None
        if negative_context is not None:
            neg_emb = self.context_matrix(negative_context)  # (BatchSize, NumNegs, EmbedDim)

        return tgt_emb, ctx_emb, neg_emb

    def compute_loss(self, input_embeds: torch.Tensor, output_embeds: torch.Tensor,
                     negative_embeds: Optional[torch.Tensor] = None):
        """
        Calculates negative sampling loss.
        """
        # True context loss
        dot_positive = torch.sum(input_embeds * output_embeds, dim=1)
        loss_pos = -F.logsigmoid(dot_positive)

        # Negative samples loss
        loss_neg = 0.0
        if negative_embeds is not None:
            # Efficient batch matrix multiplication for dot product
            # input_embeds.unsqueeze(2): (BatchSize, EmbedDim, 1)
            # negative_embeds: (BatchSize, NumNegs, EmbedDim)
            dot_negative = torch.bmm(negative_embeds, input_embeds.unsqueeze(2)).squeeze(2)
            # log(1 - sigmoid(x)) == logsigmoid(-x)
            loss_neg = -torch.sum(F.logsigmoid(-dot_negative), dim=1)

        # Final average objective
        total_loss = torch.mean(loss_pos + loss_neg)
        return total_loss

    def pred(self, input_tokens: torch.Tensor):
        """
        Retrieves final vectors for inference (sums subwords together).
        """
        encoded = self.target_matrix(input_tokens)
        if encoded.dim() == 3:
            return torch.sum(encoded, dim=1)
        return encoded

    def train_model(self, dataloader, epochs: int, neg_count: int = 10,
                    neg_prob_dist: torch.Tensor = None, eval_set=None,
                    on_epoch_end=None):
        """
        Train loop encapsulated in the model handling batches and backprop.
        """
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        # Cosine Annealing learning rate schedule
        max_iters = len(dataloader) * epochs
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

        gpu_dist = neg_prob_dist.to(ACTIVE_DEVICE) if neg_prob_dist is not None else None

        self.train()
        for epoch_idx in range(1, epochs + 1):
            epoch_loss_sum = 0.0
            
            progress = tqdm(dataloader, desc=f"Training Epoch {epoch_idx}/{epochs}")

            for step, (batch_targets, batch_contexts) in enumerate(progress):
                batch_targets = batch_targets.to(ACTIVE_DEVICE, non_blocking=True)
                batch_contexts = batch_contexts.to(ACTIVE_DEVICE, non_blocking=True)
                
                b_size = batch_targets.shape[0]

                # Sample negatives randomly using multinomial or uniform distribution
                if gpu_dist is not None:
                    negatives = torch.multinomial(gpu_dist, b_size * neg_count, replacement=True)
                    negatives = negatives.view(b_size, neg_count)
                else:
                    negatives = torch.randint(2, self.context_vocab_size, (b_size, neg_count), device=ACTIVE_DEVICE)

                optimizer.zero_grad(set_to_none=True)
                
                t_vecs, c_vecs, n_vecs = self.forward(batch_targets, batch_contexts, negatives)
                loss_val = self.compute_loss(t_vecs, c_vecs, n_vecs)
                
                loss_val.backward()
                optimizer.step()
                lr_scheduler.step()

                loss_scalar = loss_val.item()
                epoch_loss_sum += loss_scalar
                progress.set_postfix({"Loss": f"{loss_scalar:.4f}"})

            print(f"> Epoch {epoch_idx} completed. Mean Loss: {epoch_loss_sum / len(dataloader):.4f}")

            if eval_set:
                self._verify_isolation(eval_set, epoch_idx)

            if on_epoch_end:
                on_epoch_end(epoch_idx)

    @torch.no_grad()
    def _verify_isolation(self, eval_pairs, epoch_num: int):
        """Checks Spearman on the dev subset."""
        is_training_mode = self.training
        self.eval()

        vectors_A = self.pred(eval_pairs.word1_ids.to(ACTIVE_DEVICE)).cpu()
        vectors_B = self.pred(eval_pairs.word2_ids.to(ACTIVE_DEVICE)).cpu()

        cosine_sims = get_similarity_scores(vectors_A, vectors_B)
        ground_truth = pd.read_csv("data/isolated_similarity/isolated_dev_y.csv")
        spearman_corr = compute_spearman_correlation(cosine_sims, ground_truth["sim"].values)
        
        print(f"> Validation Spearman rank correlation @ epoch {epoch_num} = {spearman_corr:.5f}")

        if is_training_mode:
            self.train()


# ----------------------------------------------------
# WordNet retrofitting post-processing mechanism
# ----------------------------------------------------

def generate_synonym_graph(mapping: dict):
    """
    Extracts synsets from WordNet to form a local graph for topological retrofitting.
    Excludes padding, unk, and subwords.
    """
    print(">> Discovering WordNet adjacencies...")
    
    # Filter for standard words denoted by <xyz>
    primary_words = {
        word: ID 
        for word, ID in mapping.items()
        if word not in ("<PAD>", "<UNK>") and word.startswith("<") and word.endswith(">") and len(word) > 2
    }

    syn_graph = {}
    edges_created = 0

    for idx, (token_str, vocab_id) in enumerate(primary_words.items()):
        if (idx + 1) % 15_000 == 0:
            print(f"   Processed {idx+1} tokens, mapped {edges_created} synonyms.")
            
        plain_word = token_str[1:-1]
        matches = set()
        
        for synst in wordnet.synsets(plain_word):
            for lemma in synst.lemmas():
                formatted_lemma = f"<{lemma.name().lower()}>"
                if formatted_lemma in primary_words and formatted_lemma != token_str:
                    matches.add(primary_words[formatted_lemma])
                    
        if matches:
            syn_graph[vocab_id] = matches
            edges_created += 1

    print(f">> Synonym associations found for {edges_created} words.")
    return syn_graph


def apply_wordnet_retrofitting(vocab_indexer: dict, initial_embeds: torch.Tensor, sweeps: int = 10) -> torch.Tensor:
    """
    Adjusts standard word embeddings to reside closer to their WordNet synonyms.
    Re-distributes embeddings by calculating moving averages iteratively.
    """
    connection_graph = generate_synonym_graph(vocab_indexer)
    if not connection_graph:
        print(">> No vocabulary intersection with WordNet. Returning original matrix.")
        return initial_embeds

    matrix_device = initial_embeds.device
    
    dynamic_weights = initial_embeds.clone()
    anchor_weights = initial_embeds.clone()

    print(f">> Initiating {sweeps} WordNet retrofit sweeps...")
    for iter_step in range(1, sweeps + 1):
        temp_state = dynamic_weights.clone()
        
        for root_id, connected_ids in connection_graph.items():
            tensor_indices = torch.tensor(list(connected_ids), dtype=torch.long, device=matrix_device)
            # Sum neighbors
            adjacent_sum = torch.sum(dynamic_weights[tensor_indices], dim=0)
            
            # Weighted normalization
            temp_state[root_id] = (anchor_weights[root_id] + adjacent_sum) / (1 + len(connected_ids))
            
        dynamic_weights = temp_state
        print(f"   [Retrofit Iteration {iter_step}/{sweeps}]")

    print(">> Retrofitting applied.")
    return dynamic_weights


def export_vectors(filepath: str, wordlist, matrix_t: torch.Tensor):
    """Saves word vectors in expected line-by-line format."""
    np_mat = matrix_t.detach().cpu().numpy()
    with open(filepath, "w", encoding="utf-8") as file:
        for text_token, numeric_vec in zip(wordlist, np_mat):
            joined_vals = " ".join(f"{float_val:.6f}" for float_val in numeric_vec)
            file.write(f"{text_token} {joined_vals}\n")


# ----------------------------------------------------
# Argparse Setup and Main Runtime Flow
# ----------------------------------------------------

def retrieve_arguments():
    cl_parser = argparse.ArgumentParser(description='PyTorch Skip-Gram Configurator')
    cl_parser.add_argument('-a', '--additional_data', action='store_true',
                        help='Enable inclusion of external shards.')
    cl_parser.add_argument('-d', '--embed_dim', type=int, default=200,
                        help='Latent embedding vector dimensions.')
    cl_parser.add_argument('-w', '--window_size', type=int, default=5,
                        help='Text sequence span context perimeter.')
    cl_parser.add_argument('-wt', '--window_type', type=str, default='linear',
                        help='Scaling method within the window.')
    cl_parser.add_argument('-n', '--num_neg_samples', type=int, default=10,
                        help='Quantity of noise samples appended per truth.')
    cl_parser.add_argument('-ne', '--neg_exponent', type=float, default=0.75,
                        help='Scale applied to the multinomial negative distribution.')
    cl_parser.add_argument('--min-freq', type=int, default=5,
                        help='Token occurrence floor cutoff.')
    cl_parser.add_argument('-ep', '--num_epochs', type=int, default=3,
                        help='Full corpus pass count.')
    cl_parser.add_argument('-e', '--experiment_name', type=str, default='testing',
                        help="Sub-directory for logging current artifacts.")
    cl_parser.add_argument('--skip-retrofit', action='store_true',
                        help='Bypass semantic enforcement post processing.')

    return cl_parser.parse_args()


def main():
    config = retrieve_arguments()
    
    print(f"Target Compute Node: {ACTIVE_DEVICE}")

    # Fetch processed loaders
    train_dataset, train_dataloader, dev_set_object, test_set_object = load_data_word2vec(
        min_freq=config.min_freq, 
        window_size=config.window_size, 
        batch_size=4096
    )

    t_vocab = train_dataset.vocab_size
    c_vocab = train_dataset.context_vocab_size

    net = Word2vec(t_vocab, config.embed_dim, num_ctx_tokens=c_vocab).to(ACTIVE_DEVICE)

    distribution_priors = getattr(train_dataset, "neg_weights", None)

    # Scaffolding output hierarchies
    export_path = os.path.join("results", config.experiment_name)
    os.makedirs(export_path, exist_ok=True)
    testing_data_csv = pd.read_csv("data/isolated_similarity/isolated_test_x.csv")

    def perform_epoch_checkpoint(epoch_num):
        net.eval()
        with torch.no_grad():
            w1_inference = net.pred(test_set_object.word1_ids.to(ACTIVE_DEVICE)).cpu()
            w2_inference = net.pred(test_set_object.word2_ids.to(ACTIVE_DEVICE)).cpu()
            
        export_vectors(os.path.join(export_path, f"epoch{epoch_num}_w1.txt"),
                          testing_data_csv["word1"].values, w1_inference)
        export_vectors(os.path.join(export_path, f"epoch{epoch_num}_w2.txt"),
                          testing_data_csv["word2"].values, w2_inference)
                          
        print(f" -> Checkpointed evaluation vectors for epoch {epoch_num} inside {export_path}/")
        net.train()

    # Initiate ML routine
    net.train_model(
        dataloader=train_dataloader, 
        epochs=config.num_epochs, 
        neg_count=config.num_neg_samples,
        neg_prob_dist=distribution_priors, 
        eval_set=dev_set_object,
        on_epoch_end=perform_epoch_checkpoint
    )

    # Evaluate WordNet Semantic Retrofitting Post Step
    if not config.skip_retrofit:
        print("\n=== Running Lexical Graph Alignment via WordNet ===")
        net.target_matrix.weight.data = apply_wordnet_retrofitting(
            vocab_indexer=train_dataset.token_to_id,
            initial_embeds=net.target_matrix.weight.data,
            sweeps=10
        )

    # Perform Final Inferences
    net.eval()
    with torch.no_grad():
        final_dev_1 = net.pred(dev_set_object.word1_ids.to(ACTIVE_DEVICE))
        final_dev_2 = net.pred(dev_set_object.word2_ids.to(ACTIVE_DEVICE))
        final_test_1 = net.pred(test_set_object.word1_ids.to(ACTIVE_DEVICE))
        final_test_2 = net.pred(test_set_object.word2_ids.to(ACTIVE_DEVICE))

    # Commit submission assets
    export_vectors("results/word2vec_isol_test_words1_embeddings.txt",
                      testing_data_csv["word1"].values, final_test_1)
    export_vectors("results/word2vec_isol_test_words2_embeddings.txt",
                      testing_data_csv["word2"].values, final_test_2)
                      
    export_vectors(os.path.join(export_path, "final_w1.txt"),
                      testing_data_csv["word1"].values, final_test_1)
    export_vectors(os.path.join(export_path, "final_w2.txt"),
                      testing_data_csv["word2"].values, final_test_2)

    # Generate similarities
    sims_dev = get_similarity_scores(final_dev_1.cpu(), final_dev_2.cpu())
    sims_test = get_similarity_scores(final_test_1.cpu(), final_test_2.cpu())

    y_dev_csv = pd.read_csv("data/isolated_similarity/isolated_dev_y.csv")
    rho_dev_metric = compute_spearman_correlation(sims_dev, y_dev_csv["sim"].values)

    pd.DataFrame({"id": y_dev_csv["id"], "predicted": sims_dev}).to_csv(
        "dev_predictions.csv", index=False)
    pd.DataFrame({"id": testing_data_csv["id"], "predicted": sims_test}).to_csv(
        "test_predictions.csv", index=False)

    print("\n[FINISH] Evaluation executed over independent word pairs.")
    print(f"Validation Spearman Coefficient: {rho_dev_metric:.5f}")


if __name__ == "__main__":
    main()
