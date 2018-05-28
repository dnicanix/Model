import re
import unicodedata2
import emoji
from sklearn.feature_extraction import stop_words
import string
class PostCleaner:

    def __init__(self):
        try:
            # Wide UCS-4 build
            emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
            self.emojiDetector = re.compile('|'.join(re.escape(p) for p in emojis_list))
            self.foreignDetector = re.compile(r'[^\u0000-\u007F]')
            self.acronymDetector = re.compile(r'\s([?.!"](?:\s|$))')
            self.linkCleaner = re.compile(
                r'(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))',
                flags=re.MULTILINE)

        except re.error:
            # Narrow UCS-2 build
            self.emojiDetector = re.compile(u'(' u'\ud83c[\udf00-\udfff]|'  u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'  u'[\u2600-\u26FF\u2700-\u27BF])+',re.UNICODE)

    def normalizeUnicode(self,postContent):
        #return unicodedata2.normalize('NFKC', postContent);
        #print ('NYORKS',  'halaaaa  \u2022'.encode('ascii', 'ignore').decode("utf-8"))
        return postContent.encode('ascii', 'ignore').decode("utf-8")

    def getEmojis(self, postContent):
        #print("get emoji:",postContent)
        postContent = postContent.encode('unicode_escape')
        postContent = str(postContent, 'unicode_escape')

        for i in range(0, len(postContent)-1):
            if(self.emojiDetector.match(postContent[i]) != None and postContent[i+1] != ' '):
                postContent = postContent[:i]+self.insertSpace(postContent[i:])
        #print("result:",postContent)
        return self.emojiDetector.findall(postContent)

    def insertSpace(self,postContent):

        stop = False
        i = 0
        while(stop == False):
            if(postContent[i] != ' '):
                if (self.emojiDetector.match(postContent[i]) != None and postContent[i+1] != ' ' or
                    self.emojiDetector.match(postContent[i]) == None and self.emojiDetector.match(postContent[i+1]) != None):
                    postContent = postContent[:i+1] + ' ' + postContent[i+1:]
            if(i == len(postContent)-2):
                stop =True
            i+=1
        return postContent

    def changeLinkToText(self, postContent):
        return self.linkCleaner.sub('', postContent, count=0)

    def removeEmojis(self, postContent):
        postContent = postContent.encode('unicode_escape')
        postContent = str(postContent,'unicode_escape')

        return self.emojiDetector.sub('', postContent, count=0)

    def changeEmojisToText(self, postContent):
        return self.emojiDetector.sub('', postContent, count=0)

    def changeForeignToText(self, postContent):
        return self.foreignDetector.sub('', postContent,count=0)

    def fixAcronymSpaces(self, postContent):
        return self.acronymDetector.sub(r'\1', postContent, count=0)

    def removeExtraWhiteSpaces(self, postContent):
        """Remove extra white spaces from input text.
        This function removes white spaces from the beginning and
        the end of the string, but also duplicates white spaces
        between words.
        Args:
            text: The text to be processed.
        Returns:
            The text without extra white spaces.
        """
        return ' '.join(postContent.split())

    def removeStopWords(self, postContent, ignore_case=True):
        """Remove stop words.
        Stop words are loaded on class instantiation according
        to the specified language.
        Args:
            text: The text to be processed.
            ignore_case: Whether or not to ignore case.
            language: Code of the language to use (defaults to 'en').
        Returns:
            The text without stop words.
        """
        stopwords_dir = "Stopwords/"
        fil_stopwords = open(stopwords_dir + 'stopwords-filipino.txt','r').read().splitlines()
        additional_stopwords = open(stopwords_dir + 'stopwords-english.txt', 'r').read().splitlines()
        eng_stopwords = list(stop_words.ENGLISH_STOP_WORDS)
        return ' '.join(word for word in postContent.split(' ') if (
            word.lower() if ignore_case else word) not in (fil_stopwords + additional_stopwords + eng_stopwords))

    def replacePunctuation(self, postContent, excluded=None, replacement=''):
        """Replace punctuation symbols in text.
        Removes punctuation from input text or replaces them with a
        string if specified. Characters replaced will be those
        in string.punctuation.
        Args:
            text: The text to be processed.
            excluded: Set of characters to exclude.
            replacement: New text that will replace punctuation.
        Returns:
            The text without punctuation.
        """

        punctuations = list(string.punctuation) + list(string.digits)
        return self.replace_characters(
            postContent, characters=punctuations, replacement=replacement)

    def replace_characters(self, text, characters, replacement=''):
        """Remove characters from text.
        Removes custom characters from input text or replaces them
        with a string if specified.
        Args:
            text: The text to be processed.
            characters: Characters that will be replaced.
            replacement: New text that will replace the custom characters.
        Returns:
            The text without the given characters.
        """
        if not characters:
            return text

        characters = ''.join(sorted(characters))
        characters_regex = re.compile("[%s]" % re.escape(characters))


        return characters_regex.sub(replacement, text)