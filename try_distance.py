import time

with open('dictionnaire_fr_small.txt', 'r', encoding='utf-8') as f:
    dictio = f.read()
dictio = dictio.split('\n')

def levenshtein_distance(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m]

def get_prox_word(w):

    global dictio
    prox = None
    best_score = float('inf')

    for check in dictio:
        dist = levenshtein_distance(w, check)
        if dist < best_score:
            best_score = dist
            prox = check

    return prox

while True :
    insert = input("What word do you want to correct ? :")
    start = time.time()
    corr_word = get_prox_word(insert)
    print(f"Corrected in {time.time()-start}s !, your word is '{corr_word}' ")