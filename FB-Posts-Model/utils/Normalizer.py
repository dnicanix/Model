from utils.PostCleaner import PostCleaner

class Normalizer:
    def __init__(self):
        self.cleaned_texts = []

    def clean_text(self, texts):
        ctr = 1
        for text in texts:
            postCleaner = PostCleaner()
            # print(ctr, "----", text)
            text = bytes(text, "utf-8").decode("unicode_escape")
            text = postCleaner.changeEmojisToText(text)
            text = postCleaner.changeForeignToText(text)
            text = postCleaner.changeLinkToText(text)
            text = postCleaner.normalizeUnicode(text)
            text = postCleaner.replacePunctuation(text)
            text = postCleaner.removeExtraWhiteSpaces(text)
            text = postCleaner.removeStopWords(text)
            text = text.lower()
            self.cleaned_texts.append(text)

            # print('ctr:', ctr)
            # ctr = ctr+1

        return self.cleaned_texts


