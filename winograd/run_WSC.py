PACKAGE_PATH = './packages'
PARSER_PATH = '/project/comp5211'
WORD2VEC_PATH = './packages/word2vec/lib/python2.7/site-packages'
GENSIM_PATH = './packages/gensim/lib/python2.7/site-packages'

import sys
if not PACKAGE_PATH in sys.path:
    sys.path.append(PACKAGE_PATH)
if not PARSER_PATH in sys.path:
    sys.path.append(PARSER_PATH)
if not WORD2VEC_PATH in sys.path:
    sys.path.append(WORD2VEC_PATH)

#for input processing
import xmltodict
import itertools
from bs4 import BeautifulSoup
import re
import time

#for stanford parser
from pycorenlp import StanfordCoreNLP
import nltk
from nltk.stem.wordnet import WordNetLemmatizer

import numpy as np
from sklearn import svm, linear_model, cross_validation



#To run the Stanford Parser
#Run: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer

VERB_LIST = ['VB','VBD','VBG','VBN','VBP','VBZ']
ADJ_LIST = ['JJ','JJR','JJS']
NOUN_LIST = ['NN','NNS','NNP','NNPS']


def isVerb(x):
    if x in VERB_LIST:
        return True
    return False

def isAdj(x):
    if x in ADJ_LIST:
        return True
    return False

def isNoun(x):
    if x in NOUN_LIST:
        return True
    return False

def read_hltri_txt(filepath):
    with open(filepath) as f:
	lines = f.readlines()
    lines = [x.strip('\n') for x in lines]

    question_set=[]
    count=1
    for i in range(0,len(lines),5):
        sentence = lines[i].lower()
        target_pron = lines[i+1]
        ans1, ans2 = lines[i+2].lower().split(',')

        corr_ans=lines[i+3].lower()
        if corr_ans == ans1:
            corr_ans='A'
        elif corr_ans == ans2:
            corr_ans='B'
        else:
            print('ERR: No valid answer')
            print(sentence)
            print(ans1)
            print(ans2)
            print(corr_ans)

	key = ['idx', 'sentence', 'target_pron', 'A', 'B', 'corr_ans']
	question_set.append(dict(zip(key, [count, sentence, target_pron, ans1, ans2, corr_ans])))
	count+=1


    return question_set



def read_wsc_xml(filepath):
    with open(filepath) as f:
        doc = xmltodict.parse(f.read())

    # extract and combine sentence
    schema = [elem for elem in doc['collection']['schema']]
    sentence = [x['text']['txt1'] + ' ' + x['text']['pron'] + ' ' + x['text']['txt2'] for x in schema]

    # connect sentences with 'and' connective
    sentence = [s[:-1].replace('.', ' and').lower() for s in sentence]


    target_pron = [x['quote']['pron'].lower() for x in schema]
    ans1 = [x['answers']['answer'][0].lower() for x in schema]
    ans2 = [x['answers']['answer'][1].lower() for x in schema]
    corr_ans = [x['correctAnswer'] for x in schema]

    # pack each question into a dictionary
    key = ['idx', 'sentence', 'target_pron', 'A', 'B', 'corr_ans']
    question_set = [dict(zip(key, value)) for value in zip(range(1,273), sentence, target_pron, ans1, ans2, corr_ans)]
    return question_set


#pass sentence to StanfordCoreNLP and get tokens, dependencies
def get_tokens_and_dependencies(sentence):
    nlp = StanfordCoreNLP('http://localhost:9000')
    output = nlp.annotate(sentence, properties={'annotators': 'tokenize,ssplit,pos,depparse,parse,dcoref','outputFormat': 'json'})
    tokens = output['sentences'][0]['tokens']
    dependencies = output['sentences'][0]['basic-dependencies']
    return tokens, dependencies

# get a list of tokens of a sentence
def get_tokenized_sentence(tokens):
    tokenized_sentence = []
    for elem in tokens:
        tokenized_sentence.append(elem['word'])
    return tokenized_sentence

#get Part-of-Speech for each tokens
def get_POS(tokens):
    pos = []
    for elem in tokens:
        pos.append(elem['pos'])
    return pos


def extract_NC_features(sentence, A, B, target_pron):

    # get tokens and dependencies from stanfordNLP server
    tokens, dependencies = get_tokens_and_dependencies(sentence)

    # get sentence tokens and its part-of-speech
    sentence_tokens = get_tokenized_sentence(tokens)
    sentence_POS = get_POS(tokens)
    dict_POS = dict(zip(sentence_tokens, sentence_POS))

    # extract head word, e.g. the brown suitcase -> suitcase
    A_token = []
    B_token = []
    A_head = A
    B_head = B
    for token in sentence_tokens:
        if token in A:
            A_token.append(token)
            if isNoun(dict_POS[token]):
                A_head = token
        if token in B:
            B_token.append(token)
            if isNoun(dict_POS[token]):
                B_head = token

    # remove duplicate tokens
    A_token = set(A_token)
    B_token = set(B_token)

    # replace head phrase with head word
    text = sentence.replace(A, A_head)
    text = sentence.replace(B, B_head)

    text = str(text)
    tokens, dependencies = get_tokens_and_dependencies(text)
    dictResult = [{'rel': elem['dep'], 'gov': elem['governorGloss'],
               'dep': elem['dependentGloss'], 'dep_idx': elem['dependent']}
                 for elem in dependencies]

    # initialize all variables
    main_event = "."

    pron_role = "."
    pron_event_list = []
    pron_event_type_list = []

    A_role = "."
    A_event = "."
    B_role = "."
    B_event = "."

    flag = False
    main_candidate=[]
    for elem in dictResult:
        if (elem['rel'] in ['nsubj','dobj']) and elem['dep'] == A_head:
            A_role = elem['rel']
            main_candidate.append(elem['gov'])
        elif (elem['rel'] == 'xcomp') and elem['gov'] in main_candidate:
            main_candidate.append(elem['dep'])
        elif (elem['rel'] in ['nsubj','dobj']) and elem['dep'] == B_head and elem['gov'] in main_candidate:
            main_event = elem['gov']
            B_role = elem['rel']
            B_event = elem['gov']
            A_event = elem['gov']

        elif elem['dep'] == target_pron:
            pron_role = elem['rel']
            pron_event = elem['gov']
            if isVerb( dict_POS[pron_event] ):
                pron_event_type = 'VERB'
            elif isAdj( dict_POS[pron_event] ):
                pron_event_type = 'ADJECTIVE'
            elif isNoun( dict_POS[pron_event]):
                pron_event_type = 'NOUN'
            else:
                pron_event_type = "NA"

            pron_event_list.append( pron_event )
            pron_event_type_list.append(pron_event_type)

        elif (elem['rel'] == 'xcomp') and (elem['gov'] in pron_event_list):
            pron_event = elem['dep']
            pron_event_list.append( pron_event)
            pron_event_type_list.append(pron_event_type)

            if isVerb( dict_POS[pron_event] ):
                pron_event_type = 'VERB'
            elif isAdj( dict_POS[pron_event] ):
                pron_event_type = 'ADJECTIVE'
            elif isNoun( dict_POS[pron_event]):
                pron_event_type = 'NOUN'
            else:
                pron_event_type = "NA"

    #Find root-form of a word
    lmtzr = WordNetLemmatizer()

    main_event_root = lmtzr.lemmatize(main_event,'v')
    main_event_NC_s = main_event_root+'-s'
    main_event_NC_o = main_event_root+'-o'

    for i in range(len(pron_event_list)):
        pron_event_list[i] = lmtzr.lemmatize(pron_event_list[i],'v')


    pron_event_NC_role_list = []

    if pron_role == 'nsubj':
        for i in range(len(pron_event_list)):
            pron_event_NC_role_list.append(str(pron_event_list[i])+'-s')
    elif pron_role == 'dobj':
        for i in range(len(pron_event_list)):
            pron_event_NC_role_list.append(str(pron_event_list[i])+'-o')
    else:
        for i in range(len(pron_event_list)):
            pron_event_NC_role_list.append('null')


    #***
    #Events
    #Scores
    # -s
    # -o

    #print('===Result===')

    found = False
    f_vec = [0,0]
    NC_pair = []
    with open('./data/schemas-size12.txt') as f:
        lines = f.readlines()
        lines = [x.strip('\n') for x in lines]

        for i in range(0, len(lines)):
            token_list = lines[i].split()
            for pron_event_NC_role in pron_event_NC_role_list:
                if any ((pron_event_NC_role) in s for s in token_list):
                    if any ((main_event_NC_s) in r for r in token_list):
                        #print('found: '+main_event_NC_s+', '+pron_event_NC_role+'\n')
                        found = True
                        f_vec = [1,0]
                        NC_pair = [main_event_NC_s, pron_event_NC_role]
                    elif any ((main_event_NC_o) in r for r in token_list):
                        #print('found: '+main_event_NC_o+', '+pron_event_NC_role+'\n')
                        found = True
                        f_vec = [0,1]
                        NC_pair = [main_event_NC_o, pron_event_NC_role]


    key = ['main_event_root', 'pron_event_list', 'pron_event_type_list', 'pron_role', 'A_event', 'A_role', 'B_event', 'B_role', 'pair_found']

    tmp = [f_vec, dict(zip(key, [main_event_root, pron_event_list, pron_event_type_list, pron_role, A_event, A_role, B_event, B_role, NC_pair]))]
    #print (tmp)

    return tmp
        #if any('escape' in s for s in content[2]):
            #print(content[2])



def print_to_file(target_file, question, components, isSingle):
    #display on screen
    if isSingle == True:
        print(str(question['idx'])+': '+str(question['sentence']))
        print('main_event_root: '+ str(components['main_event_root']))
        print('A: '+ str(components['A_event']+'-'+str(components['A_role'])))
        print('B: '+ str(components['B_event']+'-'+str(components['B_role'])))
        print('pron_event_list: '+ str(components['pron_event_list']) +'-'+ str(components['pron_role']))
        print('pron_event_type_list: '+ str(components['pron_event_type_list']) +'\n')
        print('found:'+ str(components['pair_found']) + '\n\n')
    #write to file
    else:
        print>>target_file, str(question['idx'])+': '+str(question['sentence'])
        print>>target_file, 'main_event_root: '+ str(components['main_event_root'])
        print>>target_file, 'A: '+ str(components['A_event']+'-'+str(components['A_role']))
        print>>target_file, 'B: '+ str(components['B_event']+'-'+str(components['B_role']))

        print>>target_file, 'pron_event_list: '+ str(components['pron_event_list']) +'-'+ str(components['pron_role'])
        print>>target_file, 'pron_event_type_list: '+ str(components['pron_event_type_list']) +'\n'
        print>>target_file, 'found:'+ str(components['pair_found']) + '\n\n'


def test_dataset(dataset):

    isSingle_Q = False
    # Test one question
    if len(dataset) == 1:
        isSingle_Q = True
    # Test whole dataset


    corr_count = 0
    incorr_count = 0
    not_found_count = 0

    if isSingle_Q == False:
        corr_file = open('Correct.txt', 'w')
        incorr_file = open('Incorrect.txt', 'w')
        not_found_file = open('NotFound.txt', 'w')
    else:
        corr_file = -1
        incorr_file = -1
        not_found_file = -1

    print('Total sentences: '+str(len(dataset)))
    counter=0
    for question in dataset:
        if isSingle_Q == False and counter%(len(dataset)/50) == 0:
            sys.stdout.write('\r')
            sys.stdout.write("[%-50s] %d%%" % ('='*(counter/(len(dataset)/50)), 2*(counter/(len(dataset)/50))))
            sys.stdout.flush()
        counter+=1

        A = question['A']
        B = question['B']
        target_pron = question['target_pron']
        sentence = question['sentence']
        sentence = str(sentence)

        NC_vec = extract_NC_features(sentence, A, B, target_pron)

        #not found
        if NC_vec[0] == [0,0]:
            not_found_count+=1
            print_to_file(not_found_file, question, NC_vec[1], isSingle_Q)

        #correct
        elif NC_vec[0] == [1,0] and (question['corr_ans']).replace(".", "") == 'A':
            corr_count+=1
            print_to_file(corr_file, question, NC_vec[1], isSingle_Q)

        #correct
        elif NC_vec[0] == [0,1] and (question['corr_ans']).replace(".", "") == 'B':
            corr_count+=1
            print_to_file(corr_file, question, NC_vec[1], isSingle_Q)

        #incorrect
        else:
            incorr_count+=1
            print_to_file(incorr_file, question, NC_vec[1], isSingle_Q)


    print("\nCorrect Instances: "+str(corr_count))
    print("Incorrect Instances: "+str(incorr_count))
    print("Not Found: "+str(not_found_count))




#main#
if __name__ == '__main__':

    print('Reading input files..')
    hltri_questions_train = read_hltri_txt('./data/train.c.txt')
    hltri_questions_test = read_hltri_txt('./data/test.c.txt')
    wsc_questions = read_wsc_xml('./data/WSCollection.xml')



    valid = False
    while valid == False:
        print('\n===== Choose Option =====')
        print('1: Test sentence')
        print('2: Test on Existing Dataset')
        # print('3: Run SVM')
        # print('4: Try Word Similarity(Word Embedding)')
        # print('5: Try Word Similarity(Wordnet - Synsets)--Currently has bugs...')
        input_choice= raw_input('Select: ')
        if input_choice in ['1','2']:
            valid = True

    if input_choice == '1':
        input_sentence = raw_input('Enter sentence: ')
        input_pronoun = raw_input('Enter target pronoun: ')

        print('\nType Answer Choices in terms of full noun phrase (i.e. the bus driver)')
        input_A = raw_input('Enter A: ')
        input_B = raw_input('Enter B: ')
        input_Corr = raw_input('Enter Correct Answer (A/B): ')

        key = ['idx', 'sentence', 'target_pron', 'A', 'B', 'corr_ans']
        questions = []
        questions.append(dict(zip(key, [0, input_sentence, input_pronoun, input_A, input_B, input_Corr])))

        test_dataset(questions)


    if input_choice == '2':
        valid = False
        while valid == False:
            print('\n===== Choose Dataset =====')
            print('1: WSCollection')
            print('2: HLTRI_Train set')
            print('3: HLTRI_Test set')
            data_choice = raw_input('Select: ')
            if data_choice in ['1','2','3']:
                valid =  True


        if data_choice == '1':
            print('\nRunning Test on WSCollection..')
            test_dataset(wsc_questions)

        elif data_choice == '2':
            print('\nRunning Test on HLTRI_Train..')
            test_dataset(hltri_questions_train)

        elif data_choice == '3':
            print('\nRunning Test on HLTRI_Test..')
            test_dataset(hltri_questions_test)

        else:
            print("Error has occurred. Please restart")
            exit()


    # if input_choice == '4':
    #     print('\n===== Word Similarity(Word Embedding) =====')
    #     print('This program returns the cosine similarity of two words embedded in feature space trained by Google')
    #     print('Type [exit] to quit')
    #     #for word embedding
    #     import word2vec
    #     import gensim
    #
    #     while True:
    #         model = gensim.models.Word2Vec.load_word2vec_format('/project/comp5211/GoogleNews-vectors-negative300.bin', binary=True)
    #         input_word1 = raw_input('Word1: ')
    #         if input_word1 == 'exit':
    #             exit()
    #         input_word2 = raw_input('Word2: ')
    #         print('Similarity: '+model.similarity(str(input_word1), str(input_word2)))
    #
    # if input_choice == '5':
    #
    #     print('\n===== Word Similarity(WordNet - Synsets) =====')
    #     print('This program returns the path similarity of two word senses, based on the depth the two sense in the taxonomy.')
    #     print('Type [exit] to quit')
    #
    #     #https://bitbucket.org/jaganadhg/blog/src/d4f9b6091a9b/wnsim/wn-similar.py?fileviewer=file-view-default
    #     from nltk.corpus import wordnet as wn
    # 	"""
    # 	find similarity betwwn word senses of two words
    # 	"""
    #     def getSenseSimilarity(wordA,wordB):
    #     	wordA_synsets = wn.synsets(wordA)
    #     	wordB_synsets = wn.synsets(wordB)
    #     	synsetnameA = [wn.synset(str(syns.name)) for syns in wordA_synsets]
    #     	synsetnameB = [wn.synset(str(syns.name)) for syns in wordB_synsets]
    #     	for sseta, ssetb in [(sseta,ssetb) for sseta in synsetnameA for ssetb in synsetnameB]:
    #     		pathsim = sseta.path_similarity(ssetb)
    #     		wupsim = sseta.wup_similarity(ssetb)
    #     		if pathsim != None:
    #     			print ("Path Similarity: ",pathsim," WUP Sim Score: ",wupsim,"\t",sseta.definition, "\t", ssetb.definition)
    #
    #     while True:
    #         input_word1 = raw_input('Word1: ')
    #         if input_word1 == 'exit':
    #             exit()
    #         input_word2 = raw_input('Word2: ')
    #         getSenseSimilarity(str(input_word1), str(input_word2))
