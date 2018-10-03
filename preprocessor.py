
class Preprocessor():

    def __init__(self, filename):
        self.sentences = []
        ls_word = []
        ls_tags = []
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
    

    def embed_input(embedd_path, input_data = self.sentences):
        
        #change input_data into list of embedd
        #this is the path supposed to be = "data/word_embeddings.txt"
        with open(embedd_path, "r", encoding = "utf-8") as file:
            text = dict()

            for i in file:
                temp = i.split()
                text[temp[0]] = list(map(float,temp[1:]))

        #input_data in forms of = [[list of words, list of tags]. [list of words, list of tags], ... ], type list(list(list(), list()))
        for i in input_data:
            
            length = len(i[0])

            for idx in range(length):

                if i[idx] in text:
                    i[idx] = text[i[idx]]
                else:
                    i[idx] = text["<UNK_WORD>"]

        return input_data
    

    def generate_batch(self,start,limit):
        end = start + limit
        self.batch = self.sentences[start:end].copy() 
        max_length = 0
        for item in self.batch:
            if (len(item[0]) > max_length) :
                max_length = len(item[0])
            
        self.max_length = max_length

        for item in self.batch:
            while (len(item[0]) <  self.max_length):
                item[0].append('<PAD>')
                item[1].append('O')

        return self.batch
    


    def num_sentence(self):
        return len(self.sentences)
    

        
            
    