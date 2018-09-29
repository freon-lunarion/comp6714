
class Preprocessor():

    def __init__(self, filename):
        self.sentences = []
        ls_word = []
        ls_tags = []
        fl = open(filename,"r")
        
        for line in fl:
            words = line.split()
            if (not words):
                self.sentences.append((ls_word.copy(),ls_tags.copy()))
                ls_word.clear()
                ls_tags.clear()
                continue
            ls_word.append(words[0])
            ls_tags.append(words[1])
        
    def generate_batch(self,start,limit):
        end = start + limit
        self.batch = self.sentences[start:end].copy() 
        max_length = 0
        for item in self.batch:
            if (len(item[0]) > max_length) :
                max_length = len(item[0])
            
        self.max_length = max_length
    


    def num_sentence(self):
        return len(self.sentences)
    

        
            
    