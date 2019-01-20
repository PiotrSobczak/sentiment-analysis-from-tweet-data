from utils import timeit


class Preprocessor:
    sentence_len = []

    @staticmethod
    def preprocess_one(text):
        # text = text.replace("\'", "").replace("\"", "")
        text = text[1:-1]
        text = filter(text, Filters.is_invalid)
        text = text.lower()
        text = text.replace(".", " ").replace(",", " ")
        text = text.replace("!", " ! ").replace("?", " ? ")
        text = text.replace("%", " percent ")
        text = filter(text, Filters._is_empty)

        # sentence_len = len(text.split(" "))
        # Preprocessor.sentence_len.append(sentence_len)

        return text

    @staticmethod
    def preprocess_many(text_list):
        return [Preprocessor.preprocess_one(text) for text in text_list if len(text.split(" ")) < 30]


def filter(text, filter_func):
    words = text.split(" ")
    valid_words = [word for word in words if not filter_func(word)]
    return " ".join(valid_words)


class Filters:
    @staticmethod
    def _is_tag(word):
        return word.startswith("@") or word.startswith("#")

    @staticmethod
    def _is_empty(word):
        return word == "" or word == " " or word == "\t" or word == "\n";

    @staticmethod
    def _is_link(word):

        def contains(text, substring_list):
            for substr in substring_list:
                if substr in text:
                    return True
            return False

        return word.startswith(("http", "www")) or contains(word, [".net", ".com", ".org"])

    @staticmethod
    def _is_number(word):
        return word.isdigit()

    @staticmethod
    def _is_special(word):
        return word == "-" or word == "-" or word == "/" or word == "(" or word == ")"

    @staticmethod
    def is_invalid(word):
        for filter in Filters.all():
            if filter(word):
                return True
        return False

    @staticmethod
    def all():
        return [Filters._is_tag, Filters._is_link, Filters._is_number, Filters._is_empty, Filters._is_special]

