import phunspell

def correct_spelling(text):
    hobj = phunspell.Phunspell('ro_RO')
    words = text.split()
    corrected_words = []
    for word in words:
        if not hobj.lookup(word):
            suggestions = hobj.suggest(word)
            suggestion_list = list(suggestions)
            if len(suggestion_list) > 0:
                corrected_words.append(suggestion_list[0])
        else:
            corrected_words.append(word)

    corrected_text = ' '.join(corrected_words)
    print(corrected_text)  # check if text is not empty and correct
    return corrected_text
