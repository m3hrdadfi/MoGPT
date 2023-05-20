import torch
from nltk.corpus import wordnet


def word_synonym(token, sentence=None, by="None", model=None, tokenizer=None, device="cpu", topk=5):
    synonyms = []
    antonyms = []

    if by.upper() == "TRANSFORMER":
        model = model
        sent_token = tokenizer.tokenize(sentence)
        mask_index = sent_token.index(token)
        sent_token[mask_index] = tokenizer.mask_token
        indexed_tokens = tokenizer.convert_tokens_to_ids(sent_token)
        tokens_tensor = torch.tensor([indexed_tokens])
        predictions = model(tokens_tensor)[0][0][mask_index]
        synonyms = tokenizer.convert_ids_to_tokens(predictions.argsort()[-topk:].tolist()[::-1])
    else:
        for syn in wordnet.synsets(token):
            for l in syn.lemmas():
                synonyms.append(l.name())
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())

    return synonyms, antonyms
