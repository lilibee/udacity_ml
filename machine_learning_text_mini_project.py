# input mail , output stemmed mail

from nltk.stem.snowball import SnowballStemmer
import string
import re
import os
import pickle
import sys

def parseOutText(f):
    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()
    
    # generate sender field
    sender = re.findall(r"From:\s(\w+).\w+@", all_text)

    ### split off metadata
    content = all_text.split("X-FileName:")
    text_list = []
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(str.maketrans("", "",string.punctuation)
                                          )
        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
       
        stem_string = ""
        stemmer = SnowballStemmer("english") # add ignore_stopwords=True
        text_list = text_string.replace('\n', ' ').split()
 
        for word in text_list:
            stemmed_word = stemmer.stem(word)
            stem_string =  stem_string + stemmed_word + " "
          
    stemmer = SnowballStemmer("english")

    to_stem = text_string.replace('\n', ' ').split()
    words = " ".join([stemmer.stem(word) for word in to_stem])
    return stem_string, sender 



sys.path.append( "../tools/" )
# from parse_out_email_text import parseOutText

from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []
temp_counter = 0

sender = []
for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
      #  temp_counter += 1
        
        if temp_counter < 200:
           
            path = os.path.join('..', path[:-1])
           
            email = open(path, "r")
            
            parsed_email, sender = parseOutText(email)
        
            names= ['sara', 'shackleton','chris', 'germani']
            for word in names:             
                parsed_email = parsed_email.replace(word, "")

            if sender[0] == 'sara':
                from_l = 0
            elif sender[0] == 'chris':
                from_l = 1
            else:
                print("###################unknown")
                from_l = 9

            from_data.append(from_l)    
            word_data.append(parsed_email)
       # print("word_data length" ,len(word_data) )
      #  print()
    
  
      
    email.close()
    

print ("emails processed")

from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "wb") )
pickle.dump( from_data, open("your_email_authors.pkl", "wb") )


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer( sublinear_tf=True, max_df=0.5,stop_words = 'english')
fitted_vectorizer = vectorizer.fit_transform(word_data)

words = vectorizer.get_feature_names()
print(len(words), "is the number of words")
print(fitted_vectorizer.shape)

