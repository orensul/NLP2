from nltk.corpus import brown
import re
# constants
PERCENTAGE = 10
START_PSUDOING = 10
TAG = 0
COUNTS = 1
FREQ = 1
TUPLE_TAG = 1
TOTAL = 'total'
EPSILON = 0.00000000000001


PSEUDOS = [
    {'re': re.compile('^\d{2}$'), 'text': 'twoDigitNum'},
    {'re': re.compile('^\d{4}$'), 'text': 'fourDigitNum'},
    {'re': re.compile('^((\d+[a-zA-Z]+)|([a-zA-Z]+\d+))[a-zA-Z0-9]*$'),
     'text': 'containsDigitAndAlpha'},
    {'re': re.compile('^((\d+\-)|(\-\d+))\d*\-*$'),
     'text': 'containsDigitAndDash'},
    {'re': re.compile("^((\d+[\\\/])|([\\\/]\d+))[\\\/0-9]*$"),
     'text': 'containsDigitAndSlash'},
    {'re': re.compile("^((\d+\,)|(\,\d+))[\,0-9]*$"),
     'text': 'containsDigitAndComma'},
    {'re': re.compile("^((\d+\.)|(\.\d+))[\.0-9]*$"),
     'text': 'containsDigitAndPeriod'},
    {'re': re.compile("^\d+$"), 'text': 'otherNum'},
    {'re': re.compile("^[A-Z]+$"), 'text': 'allCaps'},
    {'re': re.compile("^[A-Z]\.$"), 'text': 'capPeriod'},
    {'re': re.compile("^[A-Z]\w+$"), 'text': 'capWord'},
    {'re': re.compile("^[a-z]+$"), 'text': 'lowerCase'}
]


class BrownCorpus(object):

    def __init__(self, percentage, specialPower = ''):
        # extract the last PERCENTAGE from brown corpus to be test set
        # the first 100 - PERCENTAGE from brown corpus to be the training set
        brown_news_tagged = brown.tagged_sents(categories='news')
        brown_news_tagged_size = len(brown_news_tagged)
        training_set_size = round(brown_news_tagged_size * percentage / 100)
        self.test_set = brown_news_tagged[-training_set_size:]
        self.training_set = brown_news_tagged[:brown_news_tagged_size - training_set_size]
        self.words_count = {}
        self.pseudo_count = {}
        self.pseudo_tag_count = {}
        self.tag_pseudo_count = {}
        self.tags_count = {}
        self.max_tags = {}
        self.confusion_matrix = {}
        self.viterbi_table = {}
        self.bp_table = {}
        self.known_words = set()
        self.test_words = set()
        self.unknown_words = set()
        self.tags = set()
        self.training_set_word_tag = {}
        self.training_set_tag_word = {}
        self.tag_tag_counts_dict = {}
        self.init_count()
        self.init_word_tag()
        self.init_tag_word()
        self.init_transition()
        self.init_pseudos()
        # set of unknown words = test words Minus known words
        self.unknown_words = self.test_words - self.known_words
        # initialize the dictionary training_set_word_tag such that for each tuple
        # of word and tag as a key, we will have a value which represents the count
        # (= number of occurrences) of the tag following to the word in all sentences.
        # so, for each pair, if it's the first time we encountered initialize the value
        # to be 1, otherwise, value++
        self.init_relevant_functions(specialPower)

    def init_tag_word(self):
        for sentence in self.training_set:
            for word, tag in sentence:
                if tag not in self.training_set_tag_word:
                    self.training_set_tag_word[tag] = {}
                    self.tags_count[tag] = 0
                if word not in self.training_set_tag_word[tag]:
                    self.training_set_tag_word[tag][word] = 0
                self.training_set_tag_word[tag][word] += 1
                self.tags_count[tag] += 1

    def init_word_tag(self):
        for sentence in self.training_set:
            for word, tag in sentence:
                if word not in self.training_set_word_tag:
                    self.training_set_word_tag[word] = {}
                    self.words_count[word] = 0
                if tag not in self.training_set_word_tag[word]:
                    self.training_set_word_tag[word][tag] = 0
                self.training_set_word_tag[word][tag] += 1
                self.words_count[word] += 1

    def init_count(self):
        for sentence in self.training_set:
            for word, tag in sentence:
                self.known_words.add(word)
                self.tags.add(tag)
        for sentence in self.test_set:
            for word, tag in sentence:
                self.test_words.add(word)

    def init_transition(self):
        # we can change this line of code to set biGram,
        # triGram or whatever we like :D
        for sentence in self.training_set:
            sentence += [('', 'STOP')]
            for idx in range(1, len(sentence)):
                prev_tag, tag = sentence[idx - 1][TUPLE_TAG], sentence[idx][TUPLE_TAG]
                if prev_tag not in self.tag_tag_counts_dict:
                    self.tag_tag_counts_dict[prev_tag] = {}
                if tag not in self.tag_tag_counts_dict[prev_tag]:
                    self.tag_tag_counts_dict[prev_tag][tag] = 0
                self.tag_tag_counts_dict[prev_tag][tag] += 1

    def init_pseudos(self):
        for word in self.words_count:
            if(self.words_count[word] > START_PSUDOING):
                continue
            pseudo = self.eval_pseudo_tag(word)
            if pseudo not in self.pseudo_count:
                self.pseudo_count[pseudo] = 0
                self.pseudo_tag_count[pseudo] = {}
            self.pseudo_count[pseudo] += self.words_count[word]
            for tag,count in self.training_set_word_tag[word].items():
                if tag not in self.pseudo_tag_count[pseudo]:
                    self.pseudo_tag_count[pseudo][tag] = 0
                self.pseudo_tag_count[pseudo][tag] += count
                if tag not in self.tag_pseudo_count:
                    self.tag_pseudo_count[tag] = {}
                if pseudo not in self.tag_pseudo_count[tag]:
                    self.tag_pseudo_count[tag][pseudo] = 0
                self.tag_pseudo_count[tag][pseudo] += self.words_count[word]

    def init_relevant_functions(self, specialPower):
        if (specialPower == 'PSEUDO'):
            self.emit = self.emission_pseudos
            self.transit = self.transition
        elif (specialPower == 'PLUS_ONE'):
            self.emit = self.emission_add_1_smoothing
            self.transit = self.transition_add_1_smoothing
        elif (specialPower == 'PSEUDO_PLUS'):
            self.emit = self.emission_pseudos_add_1
            self.transit = self.transition_add_1_smoothing
        elif (specialPower == ''):
            self.emit = self.emission
            self.transit = self.transition
        else:
            print("Cant find this special power please try ['PSEUDO', 'PLUS_ONE', ''] :D")
            raise ValueError




    def get_max_tag(self, word):
        """
        :param word: word to check for the most common tag.
        :return: the tag that maximize P(tag|word)
        """
        if word not in self.training_set_word_tag:
            return 'NN'
        if word not in self.max_tags:
            tags_pressed = sorted(list(self.training_set_word_tag[word].items()),
                                  key=lambda x: x[1], reverse=True)
            self.max_tags[word] = \
                tags_pressed[0][TAG]
        return self.max_tags[word]

    def emission(self, word, tag):
        """
        :param word:
        :param tag:
        :return: p(word | tag) = count(tag, word) / count(tag)
        """
        # this function will calculate p (word | tag)
        if(word == "START"):
            return 1
        if tag not in self.training_set_tag_word:
            return 0
        if word not in self.training_set_tag_word[tag]:
            return 0
        return self.training_set_tag_word[tag][word] / self.tags_count[tag]

    def emission_add_1_smoothing(self, word, tag):
        # this function will calculate p add_1(word | tag)
        num_of_tags = len(self.tags_count)
        if tag not in self.tags:
            return 1 / sum(self.tags_count.values())
        if word not in self.training_set_tag_word[tag]:
            return 1 / (self.tags_count[tag] + num_of_tags)
        return (self.training_set_tag_word[tag][word] + 1) / \
               (self.tags_count[tag] + num_of_tags)

    def emission_pseudos(self, word, tag):
        if word in self.words_count and self.words_count[word] > START_PSUDOING:
            return self.emission(word, tag)
        pseudo = self.eval_pseudo_tag(word)
        if tag not in self.tag_pseudo_count:
            return 0
        if pseudo not in self.tag_pseudo_count[tag]:
            return 0
        return self.tag_pseudo_count[tag][pseudo] / self.tags_count[tag]

    def emission_pseudos_add_1(self, word,tag):
        if word in self.words_count and self.words_count[word] > START_PSUDOING:
            return self.emission(word, tag)
        pseudo = self.eval_pseudo_tag(word)
        if tag not in self.tag_pseudo_count:
            return 1 / sum(self.tags_count.values())
        if pseudo not in self.tag_pseudo_count[tag]:
            return 1 / (self.tags_count[tag] + len(self.tags_count))
        return self.tag_pseudo_count[tag][pseudo] / self.tags_count[tag]
    def transition(self, prev_tag, tag):
        """
        :param prev_tag:
        :param tag:
        :return: count(w, v) / count(w) = q(v|w)
        """
        if prev_tag not in self.tag_tag_counts_dict:
            return 0
        if tag not in self.tag_tag_counts_dict[prev_tag]:
            return 0
        return self.tag_tag_counts_dict[prev_tag][tag] / \
               self.tags_count[prev_tag]

    def transition_add_1_smoothing(self, prev_tag, tag):
        """
        :param prev_tag:
        :param tag:
        :return: count(w, v) / count(w) = q(v|w)
        """
        if prev_tag not in self.tag_tag_counts_dict:
            return 1 / sum(self.tags_count.values())
        if tag not in self.tag_tag_counts_dict[prev_tag]:
            return 1 / self.tags_count[prev_tag] + len(self.tags_count)
        return (self.tag_tag_counts_dict[prev_tag][tag] + 1)/ (self.tags_count[prev_tag] + len(self.tags_count))

    def calculate_errors_test_set_viterbi(self):
        count_wrong = 0
        count_total = 0
        count_sentences = 0
        for sentence in self.test_set[:50]:
            count_sentences += 1
            words = ""
            tags = []
            for word, tag in sentence:
                words += word + " "
                tags.append(tag)
            viterbi_tags = self.viterbi(words)
            for i in range(len(viterbi_tags)):
                count_total += 1
                if not tags[i] == viterbi_tags[i]:
                    count_wrong += 1
                    #Confusion matrix fill
                    self.add_to_confusion(tags[i], viterbi_tags[i])
            print(count_sentences)
            print(count_wrong / count_total)
            print(tags)
            print(viterbi_tags)
        return count_wrong / count_total

    def add_to_confusion(self, orig, predict):
        if((orig, predict) not in self.confusion_matrix):
            self.confusion_matrix[(orig, predict)] = 0
        self.confusion_matrix[(orig, predict)] += 1

    def calculate_errors(self):
        """
        this function calculate the training, test and total model errors.
        and return it as a tuple
        """
        # known words error rate
        known_words_misses, known_words = 0,0
        unknown_words_misses, unknown_words = 0, 0
        for sentence in self.test_set + self.training_set:
            for word, tag in sentence:
                if word in self.training_set_word_tag:
                    known_words_misses += 1 if tag != self.get_max_tag(word) else 0
                    known_words += 1
                else:
                    unknown_words_misses += 1 if tag != 'NN' else 0
                    unknown_words += 1

        # total error
        total_error = (known_words_misses + unknown_words_misses) / \
                      (known_words + unknown_words)

        return known_words_misses / known_words, unknown_words_misses / unknown_words, \
               total_error




    def compute_max_prob(self, v, k, sentence):
        curr_max = 0
        curr_tag = "NN"
        for w in self.tags:
            viterbi = self.viterbi_table[(k, w)]
            if viterbi == 0.0:
                viterbi += EPSILON
            emission = self.emit(sentence[k+1], v)
            if emission == 0.0:
                emission += EPSILON
            transition = self.transit(w, v)
            if transition == 0.0:
                transition += EPSILON
            result = viterbi * emission * transition
            if result >= curr_max:
                curr_max = result
                curr_tag = w
        return curr_max, curr_tag




    def compute_maximize_tag_first_row(self, sentence):
        curr_max = 0
        tag = "NN"
        for w in self.tags:
            prob = self.viterbi_table[(len(sentence)-1, w)] * self.transition(w, "STOP")
            if prob >= curr_max:
                curr_max = prob
                tag = w
        return tag

    def viterbi(self, sentence):
        self.viterbiTable = {}
        self.bp_table = {}
        # split the sentence according to spaces
        sentence = sentence.split(" ")
        sentence = ["START"] + sentence
        sentence = sentence[:-1]
        n = len(sentence)
        # initialization of row 0 in viterbi table
        # initialization of row 0 in bp table

        for tag in self.tags:
            self.viterbi_table[(0, tag)] = 1

        self.bp_table[(0, "START")] = "START"

        for k in range(1, n):
            for curr_tag in self.tags:
                prob, maximize_tag = self.compute_max_prob(curr_tag, k-1, sentence)
                # if(prob > 0):
                    # print(prob, maximize_tag)
                self.viterbi_table[(k, curr_tag)] = prob
                self.bp_table[(k, curr_tag)] = maximize_tag

        tags_of_sentence = [''] * n
        tags_of_sentence[-1] = self.compute_maximize_tag_first_row(sentence)


        for k in range(n-2, 0, -1):
            tags_of_sentence[k] = self.bp_table[(k+1, tags_of_sentence[k+1])]
        return tags_of_sentence[1:]

    def print_training_tag_word_dict(self):
        print(self.training_set_tag_word)

    def print_training_word_tag_dict(self):
        print(self.training_set_word_tag)

    def print_tag_tag_counts_dict(self):
        print(self.tag_tag_counts_dict)
        print(len(self.tag_tag_counts_dict))

    def print_tags(self):
        print(self.tags)
        print(len(self.tags))

    def print_tags_count(self):

        print(sorted(list(self.tags_count.items()),
                                  key=lambda x: x[1], reverse=True))

    def print_words_count(self):
        print(sorted(list(self.words_count.items()),
                                  key=lambda x: x[1], reverse=True))


    def eval_pseudo_tag(self,word):
        for pseudoGroup in PSEUDOS:
            if(pseudoGroup['re'].match(word)):
                return pseudoGroup['text']
        return 'other'








def main():
    # initialize brown corpus training set and test set, test data will be the last
    # PERCENTAGE
    bc = BrownCorpus(PERCENTAGE, '')
    # return a list such that for each word we will have the most common tag and the
    # probability of p(tag|word)
    # bc.get_list_most_s
    # print(bc.viterbi("But Holmes was rejected again '' on the basis of his record and interview '' ."))
    # print(bc.viterbi("I want to eat"))
    print(bc.calculate_errors())
    print(bc.calculate_errors_test_set_viterbi())
    confusion = sorted(bc.confusion_matrix.items(), key=lambda x:x[1], reverse=True)
    print(confusion[0:10])
    # for sent in bc.test_set[1:150]:
    #     sent = sent[0:]
    #     sen = ' '.join([x[0] for x in sent])
    #     print('--------------------------')
    #     print(' '.join(bc.viterbi(sen)))
    #     print(' '.join([x[1] for x in sent]))
    #     print(sen)
    # print("--->")
    # bc.eval_pseudo_tag("Hello")



    #print("Known err: %s\nUnknown err: %s\nTotal err: %s"%bc.calculate_errors())


main()