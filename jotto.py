import random
import pandas as pd
import sys
import numpy as np
import string


def get_words_of_length(n):
    with open("ospd.txt") as f:
        word_list = f.read().split()
    return [x for x in word_list if len(x) == n]


def jotto_score(solution, guess):
    solution = solution.lower()
    guess = guess.lower()
    # number letters right and in right place
    num_right_place = 0
    for s, g in zip(solution, guess):
        if s == g:
            num_right_place += 1
    # number letters right (including in wrong place)
    num_right_letter = 0
    for letter in set(solution):
        num_right_letter += min(
            solution.count(letter),
            guess.count(letter))
    return num_right_letter, num_right_place


def jotto(solution):
    guess = None
    count = 0
    while guess != solution:
        count += 1
        guess = yield count
        yield jotto_score(solution, guess)


def play(legal_words, solution, strategy):
    num_guesses = 0
    while True:
        guess = strategy(legal_words)
        num_guesses += 1
        response = jotto_score(solution, guess)
        if response[1] == len(solution):
            break
        # Since this guess wasn't right, remove it from the list
        legal_words.remove(guess)
        # Remove all words that would have yielded different results
        for word in legal_words:
            if jotto(word, guess) != response:
                legal_words.remove(word)
    return num_guesses


def compete(strategies, num_games=1_000):
    legal_words = get_words_of_length(5)
    wins = {s.__name__: 0 for s in strategies}
    for ii in range(num_games):
        solution = random.choice(legal_words)
        guesses = []
        for strategy in strategies:
            num_guesses, final_answer = strategy(
                jotto(solution), legal_words)
            if final_answer != solution:
                num_guesses = sys.maxsize  # a really big number
            guesses.append(num_guesses)
        winner = strategies[guesses.index(min(guesses))].__name__
        wins[winner] += 1
    return wins


def guess_first(game, legal_words):
    possible_solutions = legal_words.copy()
    for round_num in game:
        # Guess randomly from the remaining possible solutions
        guess = possible_solutions[0]
        score = game.send(guess)
        # Since this guess wasn't right, remove it from the list
        possible_solutions.remove(guess)
        # Remove all words that would have yielded different results
        for word in possible_solutions:
            if jotto_score(word, guess) != score:
                possible_solutions.remove(word)
    return round_num, guess


def guess_random(game, legal_words):
    possible_solutions = legal_words.copy()
    for round_num in game:
        # Guess randomly from the remaining possible solutions
        guess = random.choice(possible_solutions)
        score = game.send(guess)
        # Since this guess wasn't right, remove it from the list
        possible_solutions.remove(guess)
        # Remove all words that would have yielded different results
        for word in possible_solutions:
            if jotto_score(word, guess) != score:
                possible_solutions.remove(word)
    return round_num, guess


def guess_sampled_minimax(game, legal_words):
    possible_solutions = legal_words.copy()
    for round_num in game:
        # Choose a random set of tractable size
        test_words = random.sample(possible_solutions,
                                   min(20, len(possible_solutions)))
        # Predict the jotto responses for those scored as 1 for each
        # letter correct plus 1 more for those in the right spot
        data = np.array([[list(x) for x in test_words]])
        right_place = (data == data.transpose((1, 0, 2))).sum(2)
        data = np.array([[[word.count(letter) for letter in
                         string.ascii_lowercase] for word in test_words]])
        right_letter = (np.minimum(data, data.transpose((1, 0, 2)))).sum(2)
        score = right_place + right_letter
        # Select the word with the maximum lowest possible score
        guess = test_words[score.min().argmax()]
        score = game.send(guess)

        # Since this guess wasn't right, remove it from the list
        possible_solutions.remove(guess)
        # Remove all words that would have yielded different results
        for word in possible_solutions:
            if jotto_score(word, guess) != score:
                possible_solutions.remove(word)
    return round_num, guess


def guess_minimax(game, legal_words):
    possible_solutions = legal_words.copy()

    data = np.array([[list(x) for x in possible_solutions]])
    right_place = (data == data.transpose((1, 0, 2))).sum(2)
    data = np.array([[[word.count(letter) for letter in string.ascii_lowercase]
                      for word in possible_solutions]])
    right_letter = (np.minimum(data, data.transpose((1, 0, 2)))).sum(2)
    score = right_place + right_letter
    chart = pd.DataFrame(score,
                         index=possible_solutions,
                         columns=possible_solutions)

    for round_num in game:
        # Indexing here is much faster than modifying the table
        guess = chart.loc[possible_solutions, possible_solutions].min().idxmax()
        score = game.send(guess)

        # Since this guess wasn't right, remove it from the list
        possible_solutions.remove(guess)
        # Remove all words that would have yielded different results
        for word in possible_solutions:
            if jotto_score(word, guess) != score:
                possible_solutions.remove(word)
    return round_num, guess


if __name__ == "__main__":
    from operator import itemgetter
    strategies = [
                  guess_first,
                  guess_random,
                  guess_sampled_minimax,
                 ]
    results = compete(strategies, num_games=30)
    results = sorted(results.items(), key=itemgetter(1), reverse=True)
    for strat, wins in results:
        s = "s" if wins != 1 else None
        print(f"{strat} won {wins:,.0f} time" + s)
