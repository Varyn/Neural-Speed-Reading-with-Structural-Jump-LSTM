# -*- coding: ISO-8859-1 -*-
import spacy
import pandas as pd
import numpy as np
import re
import time
import pickle
import datetime
import itertools

DEBUG = False

def processTexts(name, train, test, trainidx, testidx, padder=None):
    '''
    
    steps: 1) split into sentences
    2) remove unwanted chars (easier: keep wanted ones)
    3) split into tokens
    4) combine to single token array
    5) create token/int dictionary
    6) convert arrays to their id's

    We want time add arrays with:
    1) next comma
    2) next sentence


    '''
    train_x, train_y = train[:, trainidx[1]], train[:, trainidx[0]].astype(int)
    test_x, test_y = test[:, testidx[1]], test[:, testidx[0]].astype(int)
    print(name)
    len_train = len(train_y)

    all_text = np.concatenate((train_x, test_x))
    all_text = [re.sub(' +',' ', text).lower().replace("-","").replace("'","") for text in all_text]
    
    if padder is not None:
        print(np.mean([len(v.split(" ")) for v in all_text]))
        padtoken = "padtokenpad"
        for i in range(len(all_text)):
            leni = len(all_text[i].split(" "))
            stri = all_text[i]
            cutoff = 374
            if leni < cutoff:
                padding = " ".join([padtoken for _ in range(cutoff-leni)])
                stri = padding + stri

            all_text[i] = stri
        print(np.mean([len(v.split(" ")) for v in all_text]))

    #exit()


    nlp = spacy.load('en')#, disable=['tagger'])
    nlp.disable_pipes('ner')
    nlp.disable_pipes('tagger')

    processed_text = []
    commas = []
    newsents = []

    iii = 0
    start = time.time()
    print("START PROCESSING")
    for doc in nlp.pipe(all_text, n_threads=-1):
        iii += 1
        if iii % 1000 == 0:
            print(iii, '/', len(all_text), "docs per sec:", iii/(time.time()-start))
        sents = [s for s in doc.sents]
        
        orgtext = []
        fulltext = []
        fullcomma = []
        fullnewsent = []

        for current in sents:
            val = ([re.sub("[^a-zA-Z0-9\, ]", "", t.text.strip()) for t in current])
            val = [v for v in val if len(v) > 0]
            #print(val)
            current_comma = []
            current_val = []
            v = 1
            for i in np.arange(len(val))[::-1]:
                if "," in val[i] or ";" in val[i]:
                    v = 1
                else:
                    current_comma.append(v)
                    current_val.append(val[i])
                    v += 1
            orgtext.append(val)
            current_comma = current_comma[::-1]
            current_val = current_val[::-1]
            current_newsent = (np.arange(len(current_val))+1)[::-1].tolist()

            fulltext += current_val
            fullcomma += current_comma
            fullnewsent += current_newsent

            #print(val)
            #print(current_comma)
            #print(current_newsent)
            #print(current_val)
            #print("----")

        if DEBUG:
            print(doc)
            print(orgtext)
            print(fulltext)
            print(fullcomma)
            print(fullnewsent)
            print("-------\n\n")

        processed_text.append(fulltext)
        commas.append(fullcomma)
        newsents.append(fullnewsent)

    # dictionary time
    print("combining texts", datetime.datetime.now().time())
    onelist = list(itertools.chain.from_iterable(processed_text))

    print("total words:", len(onelist))
    print("finding unique words", datetime.datetime.now().time())
    onelist = list(set(onelist))
    
    print("total sents:", len(processed_text))
    print("making transdic", datetime.datetime.now().time())
    transDic = {onelist[i]:(i+1) for i in range(len(onelist))} # 0 is reserved for something...

    print("making reverse trans dic", datetime.datetime.now().time())
    revTransDic = {(i+1):onelist[i] for i in range(len(onelist))}

    print("total unique words:", len(transDic))

    # translate
    print("using transdic on texts", datetime.datetime.now().time())
    processed_text = [[transDic[word] for word in pp] for pp in processed_text]

    print("saving", datetime.datetime.now().time())
    # save everything
    trainstuff = [processed_text[:len_train], train_y.astype(int), commas[:len_train], newsents[:len_train]]
    teststuff  = [processed_text[len_train:], test_y.astype(int), commas[len_train:], newsents[len_train:]]

    pickle.dump([np.array(v) for v in trainstuff], open(name + "_train", "wb"))
    pickle.dump([np.array(v) for v in teststuff], open(name + "_test", "wb"))
    pickle.dump([transDic, revTransDic], open(name + "_dicts","wb"))
    
    print(len(train_y), np.unique(train_y), np.unique(test_y))
    for v in trainstuff:
        print(len(v))

    print(len(test_y))
    for v in teststuff:
        print(len(v))
    

# ------------------------------------------------------------------------------------------------------------

def processTexts_QA(name, data, all_text_to_tokenize, questions_tokenized, options_tokenized, trainlen, vallen, testlen):
    '''
    
    steps: 1) split into sentences
    2) remove unwanted chars (easier: keep wanted ones)
    3) split into tokens
    4) combine to single token array
    5) create token/int dictionary
    6) convert arrays to their id's

    We want time add arrays with:
    1) next comma
    2) next sentence


    '''

    all_text = data[:, 0]
    all_text = [re.sub(' +',' ', text).lower().replace("-","").replace("'","") for text in all_text]
    
    nlp = spacy.load('en')#, disable=['tagger'])
    nlp.disable_pipes('ner')
    nlp.disable_pipes('tagger')

    processed_text = []
    commas = []
    newsents = []

    iii = 0
    start = time.time()
    print("START PROCESSING")
    for doc in nlp.pipe(all_text, n_threads=-1):
        iii += 1
        if iii % 1000 == 0:
            print(iii, '/', len(all_text), "docs per sec:", iii/(time.time()-start))
        sents = [s for s in doc.sents]
        
        orgtext = []
        fulltext = []
        fullcomma = []
        fullnewsent = []

        for current in sents:
            val = ([re.sub("[^a-zA-Z0-9\, ]", "", t.text.strip()) for t in current])
            val = [v for v in val if len(v) > 0]
            #print(val)
            current_comma = []
            current_val = []
            v = 1
            for i in np.arange(len(val))[::-1]:
                if "," in val[i] or ";" in val[i]:
                    v = 1
                else:
                    current_comma.append(v)
                    current_val.append(val[i])
                    v += 1
            orgtext.append(val)
            current_comma = current_comma[::-1]
            current_val = current_val[::-1]
            current_newsent = (np.arange(len(current_val))+1)[::-1].tolist()

            fulltext += current_val
            fullcomma += current_comma
            fullnewsent += current_newsent

            #print(val)
            #print(current_comma)
            #print(current_newsent)
            #print(current_val)
            #print("----")

        if DEBUG:
            print(doc)
            print(orgtext)
            print(fulltext)
            print(fullcomma)
            print(fullnewsent)
            print("-------\n\n")

        processed_text.append(fulltext)
        commas.append(fullcomma)
        newsents.append(fullnewsent)

    # dictionary time
    print("combining texts", datetime.datetime.now().time())
    onelist = list(itertools.chain.from_iterable(all_text_to_tokenize)) + list(itertools.chain.from_iterable(questions_tokenized)) + list(itertools.chain.from_iterable(options_tokenized))

    print("total words:", len(onelist))
    print("finding unique words", datetime.datetime.now().time())
    onelist = list(set(onelist))
    
    print("total sents:", len(processed_text))
    print("making transdic", datetime.datetime.now().time())
    transDic = {onelist[i]:(i+1) for i in range(len(onelist))} # 0 is reserved for something...

    print("making reverse trans dic", datetime.datetime.now().time())
    revTransDic = {(i+1):onelist[i] for i in range(len(onelist))}

    print("total unique words:", len(transDic))

    # translate
    print("using transdic on texts", datetime.datetime.now().time())
    processed_text = [[transDic[word] for word in pp] for pp in processed_text]

    # translate question
    processed_questions = [[transDic[word] for word in pp] for pp in questions_tokenized]

    # translate options
    processed_options = []
    for row in options_tokenized:
        tmp = [transDic[vv] for vv in row]
        if len(tmp) != 10:
            print(row)
            exit(-1)
        #for option in row[2:-1]:
        #    tmp.append(transDic[option])
        processed_options.append(tmp)

    print("saving", datetime.datetime.now().time())

    # save everything
    #trainstuff = [processed_text[:len_train], train_y, commas[:len_train], newsents[:len_train]]
    #teststuff  = [processed_text[len_train:], test_y, commas[len_train:], newsents[len_train:]]

    trainstuff = [processed_text[:trainlen], data[:trainlen, -1].astype(int), commas[:trainlen], newsents[:trainlen], processed_questions[:trainlen], processed_options[:trainlen]]

    valstuff = [processed_text[trainlen:(trainlen+vallen)], data[trainlen:(trainlen+vallen), -1].astype(int), commas[trainlen:(trainlen+vallen)], newsents[trainlen:(trainlen+vallen)], 
                processed_questions[trainlen:(trainlen+vallen)], processed_options[trainlen:(trainlen+vallen)]]

    teststuff = [processed_text[(trainlen+vallen):], data[(trainlen+vallen):, -1].astype(int), commas[(trainlen+vallen):], newsents[(trainlen+vallen):], processed_questions[(trainlen+vallen):],
                 processed_options[(trainlen+vallen):]]

    pickle.dump([np.array(v) for v in trainstuff], open(name + "_train", "wb"))
    pickle.dump([np.array(v) for v in valstuff], open(name + "_val", "wb"))
    pickle.dump([np.array(v) for v in teststuff], open(name + "_test", "wb"))
    pickle.dump([transDic, revTransDic], open(name + "_dicts","wb"))
    '''
    print(len(train_y))
    for v in trainstuff:
        print(len(v))

    print(len(test_y))
    for v in teststuff:
        print(len(v))
    '''

def process_QA_file(path):
    with open(path) as f:
        content = f.readlines()

    data = [content[i*22:(i*22+21)] for i in range(int(len(content)/22))]
    newdata = []

    for datum in data:
        alltext = " ".join([v[2:] for v in datum[:-1]])

        lastline = datum[-1].split("\t")

        question = lastline[0][2:]
        answer = re.sub(' +',' ', lastline[1].lower()).lower().replace("-","").replace("'","") 
        options = lastline[3].lower().replace("\n","")
        options = options.split("|")
        options = [re.sub(' +',' ', text).lower().replace("-","").replace("'","") for text in options]

        answer_idx = options.index(answer)

        options = [re.sub("[^a-zA-Z0-9\, ]", "", option) for option in options]
        
        for i,v in enumerate(options):
            if len(v) == 0:
                options[i] = "0"

        if len(options) == 11: # bug in dataset, contains an additional |
            options = options[:-1]
            #print(len(options),"--",options)
            #print(datum)
            #print("\n---")


        alltext = question + " xxtekststartxx " + alltext
        newdatum = [alltext, question] + options + [answer_idx]
        #newdatum = [alltext, question] + options + [answer_idx]

        newdata.append(newdatum)

    return newdata

def process_specific_text(all_text_to_tokenize):
    all_text_to_tokenize = [re.sub(' +',' ', text).lower().replace("-","").replace("'","") for text in all_text_to_tokenize]
    
    nlp = spacy.load('en')
    nlp.disable_pipes('ner')
    nlp.disable_pipes('tagger')

    processed_text = []
    commas = []
    newsents = []

    iii = 0
    start = time.time()
    all_text_to_tokenize_indiv = []
    for doc in nlp.pipe(all_text_to_tokenize, n_threads=-1):
        iii += 1
        if iii % 1000 == 0:
            print("getting words:", iii, '/', len(all_text_to_tokenize), "docs per sec:", iii/(time.time()-start))
        sents = [s for s in doc.sents]
        
        orgtext = []
        fulltext = []
        fullcomma = []
        fullnewsent = []

        for current in sents:
            val = ([re.sub("[^a-zA-Z0-9\, ]", "", t.text.strip()) for t in current])
            val = [v for v in val if len(v) > 0]
            current_val = []
            v = 1
            for i in np.arange(len(val))[::-1]:
                if "," in val[i] or ";" in val[i]:
                    v = 1
                else:
                    current_val.append(val[i])
                    v += 1

            current_val = current_val[::-1]
            fulltext += current_val

        all_text_to_tokenize_indiv.append(fulltext)
    return all_text_to_tokenize_indiv

def doCB(name, trainpath, valpath, testpath):
    print("processing", name)

    train_data = process_QA_file(trainpath)#[:100]#[19102:19104]
    val_data = process_QA_file(valpath)#[:1]
    test_data = process_QA_file(testpath)#[:1]

    #for v in val_data:
    #    for a in v:
    #        print("##",a)
    #    print("\n----")

    combined_data = train_data + val_data + test_data
    combined_data = np.array(combined_data)

    all_text_to_tokenize = []
    answers = []
    answers_org = []
    for row in combined_data:
        if len(row) != 13:
            print("!!!",row)
            print("row len:", len(row))
            exit(-1)
        tmp = ""
        for elm in row[:-1]:
            tmp += elm + " "
        all_text_to_tokenize.append(tmp.strip())
        #all_text_to_tokenize.append(elm)
        #if "cant" == elm:
        #    print(row)
        #    print("\n ---")

        tmpoption = ""
        tmpoption_org = []
        for elm in row[2:-1]:
            tmpoption += elm + " "
            tmpoption_org.append(elm)

        answers.append(tmpoption.strip())
        answers_org.append(tmpoption_org)

    options_tokenized = answers_org #process_specific_text(answers)# #
    for i, v in enumerate(options_tokenized):
        if len(v) != 10:
            print("nr:", i, "-----", v)
            exit(-1)

    all_text_to_tokenize_indiv = process_specific_text(all_text_to_tokenize)
    questions_tokenized_indiv = process_specific_text(combined_data[:, 1])

    #for i in range(len(combined_data)):
    #    print(combined_data[i, 0])
    #    print(questions_tokenized_indiv[i])
    #    print(combined_data[i, 2:-1])
    #    print("\n----")

    trainlen = len(train_data)
    vallen = len(val_data)
    testlen = len(test_data)

    processTexts_QA(name, combined_data, all_text_to_tokenize_indiv, questions_tokenized_indiv, options_tokenized, trainlen, vallen, testlen)

    
if __name__ == '__main__':
    
    ## dbpedia
    trainpath = "dbpedia_data/dbpedia_csv/train.csv"
    testpath = "dbpedia_data/dbpedia_csv/test.csv"

    train = pd.read_csv(trainpath).values
    test = pd.read_csv(testpath).values
    trainidx = [0, 2]
    testidx = [0, 2]

    processTexts("dbpedia", train, test, trainidx, testidx)
    