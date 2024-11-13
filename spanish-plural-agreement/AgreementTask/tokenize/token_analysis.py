token_dict = []

with open('end_tokens.txt', mode='r', encoding='utf-8') as f:
    for line in f:
        line = line.split()
        token = line[1]
        if token not in token_dict:
            token_dict += [token]

print(len(token_dict))