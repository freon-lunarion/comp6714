
class Preprocessor():

    def __init__(self, filename, embedd_path):
        self.sentences = []
        self.word_embed = []
        ls_word = []
        ls_tags = []

        fl = open(embedd_path,"r", encoding = "utf8")
        for line in fl:
            words = line.split()
            self.word_embed.append(words[0])

        fl = open(filename,"r", encoding = "utf8")
        for line in fl:
            words = line.split()
            if (not words):
                self.sentences.append([ls_word.copy(),ls_tags.copy()])
                ls_word.clear()
                ls_tags.clear()
                continue
            if (words[0] in self.word_embed):
                ls_word.append(words[0])
            else :
                ls_word.append('<UNK_WORD>')
            ls_tags.append(words[1])
<<<<<<< HEAD
    

    def embed_input():
        
        #change input_data into list of embedd
        #this is the path supposed to be = "data/word_embeddings.txt"
        test = "data/word_embeddings.txt"
        with open(test, "r", encoding = "utf-8") as file:
            text = dict()

            for i in file:
                temp = i.split()
                text[temp[0]] = list(map(float,temp[1:]))

        #input_data in forms of = [[list of words, list of tags]. [list of words, list of tags], ... ], type list(list(list(), list()))
        for i in self.sentences:
            
            length = len(i[0])

            for idx in range(length):

                if i[0][idx] in text:
                    i[0][idx] = text[i[0][idx]]
                else:
                    i[0][idx] = text["<UNK_WORD>"]

        return self.sentences
    
=======
>>>>>>> 38984aaf1eeacbba59b76e6dbca9065f24f9d040

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
    

        
            
    