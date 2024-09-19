class VocabBuilder:
    @staticmethod
    def build_vocab(sentences):
        word_to_ix = {}
        for sentence in sentences:
            for word in sentence.split():
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        return word_to_ix

    @staticmethod
    def build_tag_vocab(tags):
        tag_to_ix = {}
        for tag_seq in tags:
            for tag in tag_seq:
                if tag not in tag_to_ix:
                    tag_to_ix[tag] = len(tag_to_ix)
        return tag_to_ix