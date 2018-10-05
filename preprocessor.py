
class Preprocessor():
    sentences = list()
    word_embed = dict()
    batch = list()
    embed_val = list()
    word_to_ix = {}

    def __init__(self, filename, embedd_path="data/word_embeddings.txt"):
        
        ls_word = []
        ls_tags = []

        fl = open(embedd_path,"r", encoding = "utf8")
        embed_dict = dict()
        for line in fl:
            temp = line.split()
            embed_dict[temp[0]] = list(map(float,temp[1:]))
        self.word_embed = embed_dict


        fl = open(filename,"r", encoding = "utf8")
        for line in fl:
            words = line.split()
            if (not words):
                self.sentences.append([ls_word.copy(),ls_tags.copy()])
                ls_word.clear()
                ls_tags.clear()
                continue
            
            ls_word.append(words[0])
            ls_tags.append(words[1])
    

    def embed_input(word_seq):
        
        #change input_data into list of embedd
        #this is the path supposed to be = "data/word_embeddings.txt"
        # test = "data/word_embeddings.txt"
        # with open(test, "r", encoding = "utf-8") as file:
        #     text = dict()

        #     for i in file:
        #         temp = i.split()
        #         text[temp[0]] = list(map(float,temp[1:]))

        #input_data in forms of = [[list of words, list of tags]. [list of words, list of tags], ... ], type list(list(list(), list()))
        seq = word_seq
        for i in seq:
            
            length = len(i[0])

            for idx in range(length):

                if i[0][idx] in text:
                    i[0][idx] = text[i[0][idx]]
                else:
                    i[0][idx] = text["<UNK_WORD>"]

        return seq
    

    def generate_batch(self,start=0,limit=1):
        end = start + limit
        batch = self.sentences[start:end].copy() 
        max_length = 0
        for item in batch:
            if (len(item[0]) > max_length) :
                max_length = len(item[0])
            
        self.max_length = max_length

        #padding every sentence to have same length
        for item in batch:
            while (len(item[0]) < max_length):
                item[0].append('<PAD>')
                item[1].append('O')
            
            
        for i in batch:
            
            length = len(i[0])
            for idx in range(length):

                if i[0][idx] in self.word_embed:
                    i[0][idx] = self.word_embed[i[0][idx]]
                else:
                    i[0][idx] = self.word_embed["<UNK_WORD>"]
        self.batch = batch
        return self.batch


        

    def num_sentence(self):
        return len(self.sentences)
    

        
            
    