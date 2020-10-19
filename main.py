from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import classify, NaiveBayesClassifier
from tkinter import *
import tkinter.messagebox
import nltk, os, re, string, pickle, random

'''
Preperation
'''

# Downloading the needed NLTK packages
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Get the list of stop-words
stop_words = stopwords.words('english')

classifierName = "classifier_reviews"

# Get current directory
dirCur = os.path.dirname(os.path.realpath(__file__))
dirData = dirCur+"/data"

'''
GUI
'''
def msg(text_title, text):
    tkinter.messagebox.showinfo(text_title, text)

root = Tk()
root.geometry('300x450+500+300')
root.title("Sentence sentiment analysis, adaption")

topFrame = Frame(root)
topFrame.pack(fill=X)

bottomFrame = Frame(root)
bottomFrame.pack(side=BOTTOM)

text1 = Label(topFrame, text = "Enter text:")
text1.pack()

# TextEntry
te1_frame = Frame(root)
UI_TextEntry_Main = Text(te1_frame, height=10)
ScrollBar = Scrollbar(te1_frame)
ScrollBar.config(command=UI_TextEntry_Main.yview)
UI_TextEntry_Main.config(yscrollcommand=ScrollBar.set)

# TextEntry 2
te2_frame = Frame(root)
UI_TextEntry_Main2 = Text(te2_frame, height=10,)
ScrollBar2 = Scrollbar(te2_frame)
ScrollBar2.config(command=UI_TextEntry_Main2.yview)
UI_TextEntry_Main2.config(yscrollcommand=ScrollBar2.set)

UI_Button_Convert = Button(topFrame, text = "Convert")

UI_Button_Train = Button(topFrame, text = "Re-train data")

def setUIText1(text):
    global UI_TextEntry_Main
    UI_TextEntry_Main.delete(1.0, "end")
    UI_TextEntry_Main.insert("end", text)

def setUIText2(text):
    global UI_TextEntry_Main2
    UI_TextEntry_Main2.delete(1.0, "end")
    UI_TextEntry_Main2.insert("end", text)

# Status bar
UI_Label_status_text = StringVar()
UI_Label_status = Label(root, bd=1, relief=SUNKEN, anchor=W, textvariable=UI_Label_status_text)

# New status window at bottom
def setStatus(text):
    global UI_Label_status_text
    UI_Label_status_text.set(text)
    print(text)

'''
Methods
'''

# Remove unwanted symbols (preprocessing)
def remove_noise(tokens):
    global stop_words
    cleaned_tokens = []

    for token, tag in pos_tag(tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tokens_for_model(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tokens)


# Get all tokenized files in path
def readFiles(path):
    files = os.listdir(path)

    setStatus("Reading files from " + path + "  Found " + str(len(files)) + " files.")
    ret = []
    for fname in files:
        for line in open(path+fname, 'r', encoding='utf-8'):
            ret.append(word_tokenize(line.strip()))

    return ret

# Processing, tokenizing
def prepocess(data, path):
    setStatus("Processing - " + path + ".")

    dataCleaned = []
    for tokens in data:
        dataCleaned.append(remove_noise(tokens))

    return get_tokens_for_model(dataCleaned)

'''
Load classifier
'''
classifier = None

# Creating a new calssifier, collect train, test data, prepare them, then use Bayes calssifier with them
def classifierCreate(event):
    global dirData

    positive_train_tokens_list = readFiles(dirData+"/train/pos/")
    negative_train_tokens_list = readFiles(dirData+"/train/neg/")
    positive_test_tokens_list = readFiles(dirData+"/test/pos/")
    negative_test_tokens_list = readFiles(dirData+"/test/neg/")

    positive_train_tokens_for_model = prepocess(positive_train_tokens_list, "/train/pos")
    negative_train_tokens_for_model = prepocess(negative_train_tokens_list, "/train/neg")
    positive_test_tokens_for_model = prepocess(positive_test_tokens_list, "/test/pos")
    negative_test_tokens_for_model = prepocess(negative_test_tokens_list, "/test/neg")

    positive_dataset = [(token_dict, "positive")
                            for token_dict in positive_train_tokens_for_model]

    negative_dataset = [(token_dict, "negative")
                            for token_dict in negative_train_tokens_for_model]

    positive_dataset_test = [(token_dict, "positive")
                            for token_dict in positive_test_tokens_for_model]

    negative_dataset_test = [(token_dict, "negative")
                            for token_dict in negative_test_tokens_for_model]

    dataTest = positive_dataset_test + negative_dataset_test
    dataTrain = positive_dataset + negative_dataset
    random.shuffle(dataTrain)
    random.shuffle(dataTest)

    setStatus("Training starting with "+str(len(dataTrain))+" data.")
    ret = NaiveBayesClassifier.train(dataTrain)

    accuracy = classify.accuracy(ret, dataTest)
    msg("Training finished", "Achieved accuracy (using " + str(len(dataTest)) +" training data):"+ str(accuracy))
    setStatus("Training finished")

    # print(ret.show_most_informative_features(100))

    # Save the classifier into file
    save_classifier = open(fileClassifier,"wb")
    pickle.dump(ret, save_classifier)
    save_classifier.close()

    return ret

# Load an existing classifier
def classifierLoad():
    classifier_f = open(fileClassifier, "rb")
    ret = pickle.load(classifier_f)
    classifier_f.close()
    return ret

fileClassifier = dirCur+"/"+ classifierName+".pickle"

# Check if classifier exists, if not, create one
if (os.path.exists(fileClassifier)):
    setStatus('Classifier found "' +classifierName+ '".')
    classifier = classifierLoad()
else:
    setStatus("Classifer not found, preparing data.")
    classifier = classifierCreate(None)

'''
Classification
'''
# main process
def classifyDo(text):
    global classifier

    custom_tokens = remove_noise(text)

    return classifier.classify(dict([token, True] for token in custom_tokens))

# Rekursive levenstein distance calculation
def LD(s, t):
    if s == "":
        return len(t)
    if t == "":
        return len(s)
    if s[-1] == t[-1]:
        cost = 0
    else:
        cost = 1

    res = min([LD(s[:-1], t)+1,
               LD(s, t[:-1])+1,
               LD(s[:-1], t[:-1]) + cost])
    return res

# Find word antonyms, based on levensteins distance
def getAntonym(word):
    data = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if l.antonyms():
                data.append(l.antonyms()[0].name())

    # Calculating words similarity
    distCur = 999
    ret = word
    for anonym in data:
        distNew = LD(word, anonym)
        if (distNew < distCur):
            distCur = distNew
            ret = anonym

    if ret == word:
        setStatus('Problem with word "'+ word +'" no antonyms found!')
    return ret

# Pradėti tirti tekstą
def doProcess(event):
    # text = input("\nĮveskite norimą tekstą (angliškai): ")
    text = UI_TextEntry_Main.get("1.0",END)
    text = text[:-1]

    setStatus("Checking text sentiment!")

    text_tokens = word_tokenize(text)

    result = classifyDo(text_tokens)

    setStatus('Text "' + text + '" found as ' + result + '.')

    if (result == "negative"):
        # Negative, checking which words, finding their antonyms

        textFixed = ""
        for word in text_tokens:
            spaceSymbol = " "
            if textFixed == "":
                spaceSymbol = ""

            if (classifyDo(word_tokenize(word)) == "negative"):
                antonym = getAntonym(word)

                if textFixed == "":
                    antonym = antonym.capitalize()

                textFixed += spaceSymbol + antonym
            else:
                # If there none of these, add a space symbol
                dataStop = [".", ",", "!", "?", "/", "'", '"', "[", "]", "{", "}"]

                if word in dataStop:
                    spaceSymbol = ""

                textFixed += spaceSymbol + word

        # Check if texts sentiment changed
        setUIText2(textFixed)

        setStatus("Text changed to positive.")
    else:
        setUIText2(text)
        setStatus("Niekas nepakeista, tekstas jau buvo teigiamas!")

te1_frame.pack(fill=X)
ScrollBar.pack(side=RIGHT, fill=Y)
UI_TextEntry_Main.pack(expand=YES, fill=BOTH)

UI_Button_Convert.pack()
UI_Button_Convert.bind("<Button-1>", doProcess)


te2_frame.pack(fill=X)
ScrollBar2.pack(side=RIGHT, fill=Y)
UI_TextEntry_Main2.pack(expand=YES, fill=BOTH)

UI_Button_Train.pack(side=RIGHT)
UI_Button_Train.bind("<Button-1>", classifierCreate)

UI_Label_status.pack(side=BOTTOM, fill=X)

root.mainloop()