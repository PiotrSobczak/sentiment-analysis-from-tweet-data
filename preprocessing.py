def preprocess(text):
    text = text.lower()
    return filter(text)


def filter(text):
    filters = [is_tag, is_link, is_number]
    words = text.split(" ")
    valid_words = [word for word in words if is_valid(word, filters)]
    return " ".join(valid_words)


def is_valid(word, filter_list):
    for filter in filter_list:
        if filter(word):
            return False
    return True


def is_tag(word):
    return word.startswith("@") or word.startswith("#")


def is_link(word):
    return word.startswith(("http", "www")) or contains(word, [".net", ".com", ".org"])


def is_number(word):
    return word.isdigit()


def contains(text, substring_list):
    for substr in substring_list:
        if substr in text:
            return True
    return False
