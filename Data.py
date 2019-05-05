from pprint import pprint
import nltk
import numpy as np

"""ID - Unique identifier for an example (Matches to Id in output file format)
            Text - Text containing the ambiguous pronoun and two candidate names (about a paragraph in length)
            Pronoun - The target pronoun (text)
            Pronoun-offset The character offset of Pronoun in Text
            A - The first name candidate (text)
            A-offset - The character offset of name A in Text
            B - The second name candidate
            B-offset - The character offset of name B in Text
            URL - The URL of the source Wikipedia page for the example"""

class DatasetSchema(object):
    def __init__(self, string=None):
        if string is None:
            self.id = None
            self.text = None
            self.pronoun = None
            self.pronoun_offset = None
            self.A = None
            self.A_offset = None
            self.A_coref = None
            self.B = None
            self.B_offset = None
            self.B_coref = None
            self.url = None
            self.text_length = None
        elif len(string.split("\t"))==9:
            parts = string.split("\t")
            assert True, (len(parts) == 9)
            self.id = parts[0]
            self.text = parts[1]
            self.pronoun = parts[2]
            self.pronoun_offset = int(parts[3])
            self.A = parts[4]
            self.A_offset = int(parts[5])
            self.B = parts[6]
            self.B_offset = int(parts[7])
            self.url = parts[8]
            self.text_length = self.calc_text_length(parts[1])
            self.text_numwords = self.calc_num_words(parts[1])
            self.pronoum_pos, self.a_pos, self.b_pos, self.text_tokenized, self.generalized_text = self.calc_position_words()
        else:
            parts = string.split("\t")
            assert True, (len(parts) == 11)
            self.id = parts[0]
            self.text = parts[1]
            self.pronoun = parts[2]
            self.pronoun_offset = int(parts[3])
            self.A = parts[4]
            self.A_offset = int(parts[5])
            self.A_coref = parts[6]
            self.B = parts[7]
            self.B_offset = int(parts[8])
            self.B_coref = parts[9]
            self.url = parts[10]
            self.text_length = self.calc_text_length(parts[1])
            self.text_numwords = self.calc_num_words(parts[1])
            self.pronoum_pos, self.a_pos, self.b_pos, self.text_tokenized, self.generalized_text = self.calc_position_words()


    """Getters"""
    def get_id(self):
        return self.id

    def get_text(self):
        return self.text

    def get_pronoun(self):
        return self.pronoun

    def get_pronoun_offset(self):
        return self.pronoun_offset

    def get_A(self):
        return self.A

    def get_A_offset(self):
        return self.A_offset

    def get_A_coref(self):
        return self.A_coref

    def get_B(self):
        return self.B

    def get_B_offset(self):
        return self.B_offset

    def get_B_coref(self):
        return self.B_coref

    def get_url(self):
        return self.url

    def get_text_length(self):
        return self.text_length

    def get_text_num_words(self):
        return self.text_numwords

    def get_generalized_text(self):
        return self.generalized_text
    def get_A_pos(self):
        return self.a_pos
    def get_B_pos(self):
        return self.b_pos
    def get_Pronoun_pos(self):
        return self.pronoum_pos


    def calc_position_words(self):
        arrs_offsets = [self.pronoun_offset, self.A_offset, self.B_offset]
        max_off = np.argsort(arrs_offsets)[::-1][:3]
        text = self.text
        s = False
        for offset in max_off:
            if offset == 0:
                left_part = text[:self.pronoun_offset]
                rightpart = text[self.pronoun_offset:]
                if self.pronoun in ['Her','her', 'His','his']:
                    s = True
                    rightpart_after = rightpart.replace(self.pronoun,'PPPCS', 1)
                else:
                    rightpart_after = rightpart.replace(self.pronoun,'PPPC', 1)
                text = left_part + rightpart_after
            elif offset == 1:
                left_part = text[:self.A_offset]
                rightpart = text[self.A_offset:]

                rightpart_after = rightpart.replace(self.A, 'AAAC', 1)
                text = left_part + rightpart_after
            else:
                left_part = text[:self.B_offset]
                rightpart = text[self.B_offset:]
                rightpart_after = rightpart.replace(self.B, 'BBBC', 1)
                text = left_part + rightpart_after

        final_text = text.replace(self.A, 'AAA').replace(self.B, 'BBB')
        tokens = nltk.word_tokenize(final_text)
        if s:
            try:
                pronoun_pos = tokens.index("PPPCS")
            except:
                for elem in tokens:
                    if "PPPCS" in elem:
                        pronoun_pos = tokens.index(elem)
            s=False
        else:
            try:
                pronoun_pos = tokens.index("PPPC")
            except:
                for elem in tokens:
                    if "PPPC" in elem:
                        pronoun_pos = tokens.index(elem)
        try:
            a_pos = tokens.index("AAAC")
        except:
            for elem in tokens:
                if "AAAC" in elem:
                    a_pos = tokens.index(elem)
        try:
            b_pos = tokens.index("BBBC")
        except:
            for elem in tokens:
                if "BBBC" in elem:
                    b_pos = tokens.index(elem)
        return pronoun_pos, a_pos, b_pos, tokens, final_text

    """Setters"""

    def set_id(self, devid):
        self.id = devid

    def set_text(self, text):
        self.text = text

    def set_pronoun(self, pronoun):
        self.pronoun = pronoun

    def set_pronoun_offset(self, pronoun_offset):
        self.pronoun_offset = pronoun_offset

    def set_a(self, a):
        self.A = a

    def set_a_offset(self, aoffset):
        self.A_offset = aoffset

    def set_a_coref(self, acorref):
        self.A_coref = acorref

    def set_b(self, b):
        self.B = b

    def set_b_offset(self, boffeset):
        self.B_offset = boffeset

    def set_b_coref(self, bcoref):
        self.B_coref = bcoref

    def set_url(self, url):
        self.url = url


    """"Additional Functions"""

    def calc_text_length(self, text):
        return len(text)

    def calc_num_words(self, text):
        tokens = nltk.word_tokenize(text)
        return len(tokens)


    def tokenize_text(self, text):
        tokens = nltk.word_tokenize(text)
        return tokens


    def pretty_print(self):
        pprint(vars(self))

