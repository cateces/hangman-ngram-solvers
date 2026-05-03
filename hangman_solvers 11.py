"""
Hangman Solvers
Solvers: Random, Frequency, 2-gram (backoff), 3-gram (backoff),
         Interpolation (equal), Interpolation (trigram-heavy),
         Interpolation (unigram-heavy), Pattern (candidate filtering)
Metrics: Win rate, Average mistakes, Perplexity
"""

import random
import string
import math
from collections import defaultdict


# ─────────────────────────────────────────────
# DATA LOADING & SPLITTING
# ─────────────────────────────────────────────

def load_words(filepath):
    """Load one word per line from a text file. Returns a list of lowercase words."""
    with open(filepath, "r") as f:
        words = [line.strip().lower() for line in f if line.strip().isalpha()]
    return words


def split_data(words, train_size=50000, test_size=10000, seed=42):
    """Randomly split words into training and test sets."""
    random.seed(seed)
    shuffled = words[:]
    random.shuffle(shuffled)
    train = shuffled[:train_size]
    test = shuffled[train_size:train_size + test_size]
    return train, test


# ─────────────────────────────────────────────
# GAME ENGINE
# ─────────────────────────────────────────────

def play_game(solver, word, max_mistakes=6):
    """
    Simulate one hangman game.
    - solver: a callable(revealed, guessed) -> letter
    - word: the target word (string)
    - Returns (won: bool, mistakes: int)
    """
    revealed = ["_"] * len(word)
    guessed = set()
    mistakes = 0

    while "_" in revealed and mistakes < max_mistakes:
        guess = solver(revealed, guessed)
        guessed.add(guess)

        if guess in word:
            for i, ch in enumerate(word):
                if ch == guess:
                    revealed[i] = guess
        else:
            mistakes += 1

    won = "_" not in revealed
    return won, mistakes


def compute_perplexity(prob_solver, test_words):
    """
    Compute perplexity of a probability-based solver on the test set.
    prob_solver: callable(revealed, guessed, target_letter) -> probability
    Perplexity = exp(- (1/N) * sum of log P(correct letter at each position))
    N = total number of letters across all test words.
    """
    total_log_prob = 0.0
    total_letters = 0

    for word in test_words:
        revealed = ["_"] * len(word)
        guessed = set()

        for i, true_letter in enumerate(word):
            # Get probability the solver assigns to the correct letter
            prob = prob_solver(list(revealed), guessed, true_letter)
            prob = max(prob, 1e-10)  # avoid log(0)
            total_log_prob += math.log(prob)
            total_letters += 1

            # Reveal the true letter and mark as guessed
            revealed[i] = true_letter
            guessed.add(true_letter)

    perplexity = math.exp(-total_log_prob / total_letters)
    return perplexity


def evaluate_solver(solver_fn, test_words, max_mistakes=6):
    """Run solver on all test words. Returns win rate and avg mistakes."""
    wins = 0
    total_mistakes = 0

    for word in test_words:
        won, mistakes = play_game(solver_fn, word, max_mistakes)
        if won:
            wins += 1
        total_mistakes += mistakes

    win_rate = wins / len(test_words)
    avg_mistakes = total_mistakes / len(test_words)
    return win_rate, avg_mistakes


# ─────────────────────────────────────────────
# SOLVER 1: RANDOM
# ─────────────────────────────────────────────

def make_random_solver():
    """Guesses a random letter from the full alphabet each turn (may repeat)."""
    def solver(revealed, guessed):
        return random.choice(string.ascii_lowercase)

    def prob_solver(revealed, guessed, target):
        return 1.0 / 26.0

    return solver, prob_solver


# ─────────────────────────────────────────────
# SOLVER 2: FREQUENCY-BASED
# ─────────────────────────────────────────────

def build_frequency_model(train_words):
    """Count overall letter frequencies across all training words."""
    freq = defaultdict(int)
    for word in train_words:
        for ch in word:
            freq[ch] += 1
    return freq


def make_frequency_solver(freq_model):
    """
    At each step, guess the most frequent letter (globally) that hasn't been guessed.
    Does not use revealed pattern — purely frequency ranked.
    """
    ranked = sorted(freq_model, key=lambda c: freq_model[c], reverse=True)
    total = sum(freq_model.values())

    def solver(revealed, guessed):
        for letter in ranked:
            if letter not in guessed:
                return letter
        remaining = [c for c in string.ascii_lowercase if c not in guessed]
        return random.choice(remaining) if remaining else random.choice(string.ascii_lowercase)

    def prob_solver(revealed, guessed, target):
        return freq_model.get(target, 0) / total if total > 0 else 1.0 / 26.0

    return solver, prob_solver


# ─────────────────────────────────────────────
# N-GRAM MODEL (left-to-right, backoff)
# ─────────────────────────────────────────────

def build_ngram_model(train_words, n):
    """
    Build an n-gram model from training words.
    counts[(context_tuple)] = {next_letter: count}
    context is the (n-1) letters immediately to the left of position i.
    Uses '#' as start-of-word padding symbol.
    """
    counts = defaultdict(lambda: defaultdict(int))
    for word in train_words:
        padded = "#" * (n - 1) + word
        for i in range(n - 1, len(padded)):
            context = tuple(padded[i - (n - 1):i])
            target = padded[i]
            counts[context][target] += 1
    return counts


def build_unigram_model(train_words):
    """Unigram counts — ultimate fallback for backoff and interpolation."""
    counts = defaultdict(int)
    for word in train_words:
        for ch in word:
            counts[ch] += 1
    return counts


def get_ngram_prob(letter, context, ngram_models, unigram_model, n):
    """
    Get backoff probability for a letter given left context.
    Tries n-gram, falls back through lower orders to unigram.
    """
    for order in range(n, 0, -1):
        if order == 1:
            total = sum(unigram_model.values())
            return unigram_model.get(letter, 0) / (total if total > 0 else 1)
        else:
            model = ngram_models[order]
            ctx = context[-(order - 1):]
            if ctx in model and model[ctx]:
                total = sum(model[ctx].values())
                return model[ctx].get(letter, 0) / total
    return 1.0 / 26.0


def score_letter_ngram(letter, revealed, ngram_models, unigram_model, n):
    """
    Score a candidate letter across all unknown positions using backoff n-gram context.
    """
    score = 0.0
    padded = ["#"] * (n - 1) + list(revealed)

    for i in range(n - 1, len(padded)):
        actual_i = i - (n - 1)
        if actual_i < 0 or actual_i >= len(revealed):
            continue
        if revealed[actual_i] != "_":
            continue

        raw_context = padded[i - (n - 1):i]
        context = tuple(c if c != "_" else "#" for c in raw_context)
        score += get_ngram_prob(letter, context, ngram_models, unigram_model, n)

    return score


def make_ngram_solver(ngram_models, unigram_model, n):
    """Backoff n-gram solver and its probability function."""
    def solver(revealed, guessed):
        candidates = [c for c in string.ascii_lowercase if c not in guessed]
        if not candidates:
            return random.choice(string.ascii_lowercase)
        return max(candidates, key=lambda c: score_letter_ngram(c, revealed, ngram_models, unigram_model, n))

    def prob_solver(revealed, guessed, target):
        # Average probability across all positions
        padded = ["#"] * (n - 1) + list(revealed)
        total_positions = sum(1 for ch in revealed if ch == "_")
        if total_positions == 0:
            return 1.0
        score = 0.0
        for i in range(n - 1, len(padded)):
            actual_i = i - (n - 1)
            if actual_i < 0 or actual_i >= len(revealed):
                continue
            if revealed[actual_i] != "_":
                continue
            raw_context = padded[i - (n - 1):i]
            context = tuple(c if c != "_" else "#" for c in raw_context)
            score += get_ngram_prob(target, context, ngram_models, unigram_model, n)
        return score / total_positions

    return solver, prob_solver


# ─────────────────────────────────────────────
# INTERPOLATION SOLVER
# ─────────────────────────────────────────────

def score_letter_interpolation(letter, revealed, ngram_models, unigram_model, lambdas):
    """
    Score a letter using interpolation: blend unigram, bigram, trigram probabilities.
    lambdas = (lambda1, lambda2, lambda3) for unigram, bigram, trigram.
    """
    l1, l2, l3 = lambdas
    score = 0.0
    n = 3  # max order
    padded = ["#"] * (n - 1) + list(revealed)

    for i in range(n - 1, len(padded)):
        actual_i = i - (n - 1)
        if actual_i < 0 or actual_i >= len(revealed):
            continue
        if revealed[actual_i] != "_":
            continue

        raw_context = padded[i - (n - 1):i]
        context = tuple(c if c != "_" else "#" for c in raw_context)

        # Unigram probability
        uni_total = sum(unigram_model.values())
        p_uni = unigram_model.get(letter, 0) / (uni_total if uni_total > 0 else 1)

        # Bigram probability
        bi_ctx = context[-1:]
        bi_model = ngram_models[2]
        if bi_ctx in bi_model and bi_model[bi_ctx]:
            bi_total = sum(bi_model[bi_ctx].values())
            p_bi = bi_model[bi_ctx].get(letter, 0) / bi_total
        else:
            p_bi = p_uni

        # Trigram probability
        tri_ctx = context[-2:]
        tri_model = ngram_models[3]
        if tri_ctx in tri_model and tri_model[tri_ctx]:
            tri_total = sum(tri_model[tri_ctx].values())
            p_tri = tri_model[tri_ctx].get(letter, 0) / tri_total
        else:
            p_tri = p_bi

        # Interpolated score
        score += l1 * p_uni + l2 * p_bi + l3 * p_tri

    return score


def make_interpolation_solver(ngram_models, unigram_model, lambdas):
    """Interpolation solver and its probability function."""
    def solver(revealed, guessed):
        candidates = [c for c in string.ascii_lowercase if c not in guessed]
        if not candidates:
            return random.choice(string.ascii_lowercase)
        return max(candidates, key=lambda c: score_letter_interpolation(c, revealed, ngram_models, unigram_model, lambdas))

    def prob_solver(revealed, guessed, target):
        total_positions = sum(1 for ch in revealed if ch == "_")
        if total_positions == 0:
            return 1.0
        score = score_letter_interpolation(target, revealed, ngram_models, unigram_model, lambdas)
        return score / total_positions

    return solver, prob_solver


# ─────────────────────────────────────────────
# PATTERN SOLVER (CANDIDATE FILTERING)
# ─────────────────────────────────────────────

def word_matches_pattern(word, revealed, guessed):
    """Check if a training word is compatible with the current game state."""
    if len(word) != len(revealed):
        return False
    wrong_letters = {c for c in guessed if c not in revealed}
    for ch in wrong_letters:
        if ch in word:
            return False
    for i, ch in enumerate(revealed):
        if ch != "_" and word[i] != ch:
            return False
    return True


def make_pattern_solver(train_words):
    """
    Filters training words to those matching the current pattern,
    then guesses the most frequent letter at unknown positions.
    Pre-indexed by length for efficiency.
    """
    words_by_length = defaultdict(list)
    for word in train_words:
        words_by_length[len(word)].append(word)

    global_freq = defaultdict(int)
    for word in train_words:
        for ch in word:
            global_freq[ch] += 1
    global_total = sum(global_freq.values())
    global_ranked = sorted(global_freq, key=lambda c: global_freq[c], reverse=True)

    def solver(revealed, guessed):
        length_matched = words_by_length[len(revealed)]
        candidates = [w for w in length_matched if word_matches_pattern(w, revealed, guessed)]

        letter_counts = defaultdict(int)
        unknown_positions = [i for i, ch in enumerate(revealed) if ch == "_"]

        for word in candidates:
            for i in unknown_positions:
                ch = word[i]
                if ch not in guessed:
                    letter_counts[ch] += 1

        eligible = [c for c in letter_counts if c not in guessed]
        if eligible:
            return max(eligible, key=lambda c: letter_counts[c])

        for letter in global_ranked:
            if letter not in guessed:
                return letter

        return random.choice(string.ascii_lowercase)

    def prob_solver(revealed, guessed, target):
        length_matched = words_by_length[len(revealed)]
        candidates = [w for w in length_matched if word_matches_pattern(w, revealed, guessed)]

        unknown_positions = [i for i, ch in enumerate(revealed) if ch == "_"]
        if not candidates or not unknown_positions:
            return global_freq.get(target, 0) / global_total if global_total > 0 else 1.0 / 26.0

        letter_counts = defaultdict(int)
        total_counts = 0
        for word in candidates:
            for i in unknown_positions:
                letter_counts[word[i]] += 1
                total_counts += 1

        return letter_counts.get(target, 0) / total_counts if total_counts > 0 else 1.0 / 26.0

    return solver, prob_solver


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_all(train_words):
    """Build all models from training data."""
    print("Training frequency model...")
    freq_model = build_frequency_model(train_words)

    print("Training unigram model...")
    unigram_model = build_unigram_model(train_words)

    print("Training n-gram models (2, 3)...")
    ngram_models = {}
    for n in [2, 3]:
        ngram_models[n] = build_ngram_model(train_words, n)
        print(f"  {n}-gram model trained ({len(ngram_models[n])} contexts)")

    return freq_model, unigram_model, ngram_models


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Load & split data ──
    print("Loading words from 'words_alpha.txt'...")
    all_words = load_words("/content/words_alpha.txt")
    all_words = [w for w in all_words if len(w) >= 5]
    print(f"After filtering to 5+ letter words: {len(all_words)}")

    train_words, test_words = split_data(all_words, train_size=50000, test_size=10000)
    print(f"Train: {len(train_words)} words | Test: {len(test_words)} words\n")

    # ── Train models ──
    freq_model, unigram_model, ngram_models = train_all(train_words)
    print()

    # ── Build solvers (each returns a game solver and a probability solver) ──
    solver_configs = {
        "Random":         make_random_solver(),
        "Frequency":      make_frequency_solver(freq_model),
        "2-gram":         make_ngram_solver(ngram_models, unigram_model, 2),
        "3-gram":         make_ngram_solver(ngram_models, unigram_model, 3),
        "Interp-Equal":   make_interpolation_solver(ngram_models, unigram_model, (1/3, 1/3, 1/3)),
        "Interp-Trigram": make_interpolation_solver(ngram_models, unigram_model, (0.1, 0.3, 0.6)),
        "Interp-Unigram": make_interpolation_solver(ngram_models, unigram_model, (0.6, 0.3, 0.1)),
        "Pattern":        make_pattern_solver(train_words),
    }

    # ── Evaluate ──
    results = {}
    for name, (solver, prob_solver) in solver_configs.items():
        print(f"Evaluating {name} solver on {len(test_words)} test words...")
        win_rate, avg_mistakes = evaluate_solver(solver, test_words, max_mistakes=6)
        perplexity = compute_perplexity(prob_solver, test_words)
        results[name] = (win_rate, avg_mistakes, perplexity)

    # ── Print summary table ──
    print("\n" + "=" * 68)
    print(f"{'Solver':<16} {'Win Rate':>10} {'Avg Mistakes':>14} {'Perplexity':>14}")
    print("=" * 68)
    for name, (win_rate, avg_mistakes, perplexity) in results.items():
        print(f"{name:<16} {win_rate*100:>9.1f}% {avg_mistakes:>14.2f} {perplexity:>14.2f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
