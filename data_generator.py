import pickle
import numpy as np
import os
from queue import Queue
from multiprocessing import Process
from threading import Thread
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt



class DataGenerator():
    @staticmethod
    def dataGenerator2(x_val, query, x_q_a_words, y_val, x_comma, x_newsents, batch_size, max_label, max_len=np.infty):
        perm = np.random.permutation(len(y_val))
        x_val = x_val[perm]
        y_val = y_val[perm]
        x_comma = x_comma[perm]
        x_newsents = x_newsents[perm]
        if query is not None:
            query = query[perm]
            x_q_a_words = x_q_a_words[perm]

        num_samples = len(y_val)
        # while True:
        for i in range(int(np.ceil(num_samples / batch_size))):
            fromval = batch_size * (i)
            toval = batch_size * (i + 1)

            if toval <= num_samples:
                this_y = y_val[fromval:toval]

                mask = np.ones(batch_size)
                seqs = np.array([len(v) for v in x_val[fromval:toval]])

            else:
                toval = num_samples

                this_y = np.zeros(batch_size)
                mask = np.zeros(batch_size)

                this_y[:(toval - fromval)] = y_val[fromval:toval]
                mask[:(toval - fromval)] = np.ones(toval - fromval)
                seqs = np.zeros(batch_size).astype(int)
                seqs[:(toval - fromval)] = np.array([len(v) for v in x_val[fromval:toval]])

            rows, cols = len(x_val), np.max([len(v) for v in x_val[fromval:toval]])

            #adujusting for max len:
            cols = int(np.min([cols, max_len])) + 1

            for i in range(len(seqs)):
                seqs[i]=int(np.min([seqs[i], max_len]))

            this_x = np.zeros((batch_size, cols))
            this_comma = np.zeros((batch_size, cols))
            this_newsents = np.zeros((batch_size, cols))

            if x_q_a_words is not None:
                seqs_query = np.zeros(batch_size, dtype=np.int32)
                seqs_query[:(toval - fromval)] = np.array([len(v) for v in query[fromval:toval]])
                x_q_a_words_batch = np.zeros((batch_size, 10))
                x_q_a_words_batch[:(toval - fromval)] = x_q_a_words[fromval:toval]
                cols_query = np.max([len(v) for v in query[fromval:toval]]) + 1
                query_batch = np.zeros((batch_size, cols_query))

            jcount = 0
            for j in range(fromval, toval):
                if len(x_val[j]) >= cols-1:
                    go_upto = cols-1 #minus 1 because col is 1 bigger for always having 1 padding of empty
                else:
                    go_upto = len(x_val[j])
                element = x_val[j]
                comma_element = x_comma[j]
                newsent_element = x_newsents[j]
                this_x[jcount, :go_upto] = element[:go_upto]
                this_comma[jcount, :go_upto] = comma_element[:go_upto]
                this_newsents[jcount, :go_upto] = newsent_element[:go_upto]

                if x_q_a_words is not None:
                    query_batch[jcount, :len(query[j])] = query[j]
                jcount += 1

            if x_q_a_words is not None:
                query_batch = query_batch.T


            '''
            this_x = this_x.T.reshape(cols, batch_size, 1)
            this_y = this_y.reshape(batch_size, 1)
            # print(this_comma.shape)
            this_comma = this_comma.T.reshape(cols, batch_size, 1)
            this_newsents = this_newsents.T.reshape(cols, batch_size, 1)
            mask = mask.reshape(batch_size, 1)
            '''
            this_x = this_x.transpose()
            this_y = this_y.transpose()
            this_comma = this_comma.transpose()
            this_newsents = this_newsents.transpose()

            this_addMax = cols - 1



            # print(seqs, this_addMax)
            if x_q_a_words is None:
                yield ((this_x, this_y, this_comma, this_newsents, this_addMax, mask, seqs, None), np.sum(mask==0))
            else:
                yield ((this_x, this_y, this_comma, this_newsents, this_addMax, mask, seqs, x_q_a_words_batch), np.sum(mask==0))


    @staticmethod
    def makeVal(x, y, comma, newsent, val_size, query=None, q_a_words=None):
        lenData = len(y)
        perm = np.random.permutation(lenData)
        x = x[perm]
        y = y[perm]
        comma = comma[perm]
        newsent = newsent[perm]
        if query is not None:
            query = query[perm]
            q_a_words = q_a_words[perm]

        splitAt = int(lenData * (1 - val_size))

        x_train = x[0:splitAt]
        x_val = x[splitAt:]

        y_train = y[0:splitAt]
        y_val = y[splitAt:]

        comma_train = comma[0:splitAt]
        comma_val = comma[splitAt:]

        newsent_train = newsent[0:splitAt]
        newsent_val = newsent[splitAt:]

        q_a_words_train = None
        q_a_words_val = None
        query_train = None
        query_val = None
        if q_a_words is not None:
            q_a_words_train = q_a_words[0:splitAt]
            q_a_words_val = q_a_words[splitAt:]
            query_train = query[0:splitAt]
            query_val = query[splitAt:]

        return x_train, y_train, comma_train, newsent_train, query_train, q_a_words_train, x_val, y_val, comma_val, newsent_val, query_val, q_a_words_val


    @staticmethod
    def generate_data_from_name(datasetname, folder_data, folder_result, batch_size, is_Q_A, val_size, max_len=np.infty):
        q_a_words_train = None
        q_a_words_test = None
        query_train = None
        query_test = None

        folder_and_name = folder_data + datasetname
        if is_Q_A:
            x_train, y_train, comma_train, newsent_train, query_train, q_a_words_train = pickle.load(
                open(folder_and_name + "_train", "rb"))
            x_test, y_test, comma_test, newsent_test, query_test, q_a_words_test = pickle.load(
                open(folder_and_name + "_test", "rb"))

            try:
                x_val, y_val, comma_val, newsent_val, query_val, q_a_words_val = pickle.load(
                    open(folder_and_name + "_val", "rb"))
                has_val = True
            except FileNotFoundError:
                has_val = False



        else:
            x_train, y_train, comma_train, newsent_train, = pickle.load(open(folder_and_name + "_train", "rb"))
            x_test, y_test, comma_test, newsent_test, = pickle.load(open(folder_and_name + "_test", "rb"))
            try:
                x_val, y_val, comma_val, newsent_val = pickle.load(
                    open(folder_data + folder_and_name + "_val", "rb"))
                has_val = True
            except FileNotFoundError:
                has_val = False

        # there is some noise with empty strings, remove it
        '''
        todel = []
        for i in range(len(x_train)):
            if len(x_train[i]) < 1:
                todel.append(i)

        x_train = np.delete(x_train, todel)
        y_train = np.delete(y_train, todel)
        train_comma = np.delete(comma_train, todel)
        train_newsent = np.delete(newsent_train, todel)
        '''

        if not has_val:
            split_file = folder_result + datasetname + "_train_val_split_" + str(val_size)
            if not os.path.isfile(split_file):
                x_train, y_train, comma_train, newsent_train, query_train, q_a_words_train, x_val, y_val, comma_val, \
                newsent_val, query_val, q_a_words_val = DataGenerator.makeVal(x_train, y_train, comma_train, newsent_train, val_size,
                                                                query_train, q_a_words_train)

                toSave = (
                    x_train, y_train, comma_train, newsent_train, query_train, q_a_words_train, x_val, y_val, comma_val, \
                    newsent_val, query_val, q_a_words_val)
                pickle.dump(toSave, open(split_file, "wb"))
            else:
                #print("using previusly made train/val")
                (x_train, y_train, comma_train, newsent_train, query_train, q_a_words_train, x_val, y_val, comma_val, \
                 newsent_val, query_val, q_a_words_val) = pickle.load(open(split_file, "rb"))

        transDic, revTransDic = pickle.load(open(folder_and_name + "_dicts", "rb"))

        # specific for when they use 1 indexing instead of 0 indexing
        if min(y_train) > 0:
            y_train -= 1
            y_val -= 1
            y_test -= 1

        n_words = len(transDic) + 1  # len(vocab_processor.vocabulary_)+1
        max_label = np.max(y_train) + 1
        #print(np.unique(y_train))
        # lens = [len(v) for v in x_train]

        generatorTrain = lambda: DataGenerator.dataGenerator2(x_train, query_train, q_a_words_train, y_train, comma_train,
                                                newsent_train,
                                                batch_size, max_label, max_len)
        generatorVal = lambda: DataGenerator.dataGenerator2(x_val, query_val, q_a_words_val, y_val, comma_val, newsent_val,
                                              batch_size,
                                              max_label, max_len)
        generatorTest = lambda: DataGenerator.dataGenerator2(x_test, query_test, q_a_words_test, y_test, comma_test, newsent_test,
                                               batch_size, max_label, max_len)

        return transDic, max_label, n_words, generatorTrain, generatorVal, generatorTest

    @staticmethod
    def matrix_for_embed(w2v,trans_dic):
        size = w2v["king"].shape[0]
        number_words = len(trans_dic)+1 #+1 because we 0 for masking/no word
        w2v_for_dataset = np.random.uniform(-0.05, 0.05, (number_words, size))  # np.zeros((maxId,size[0]), dtype = np.float32)
        for word in trans_dic:
            wordid = trans_dic[word]
            if word in w2v:
                w2v_for_dataset[wordid, :] = w2v[word]
        return w2v_for_dataset.astype(np.float32)


    @staticmethod
    def _worker(gen_lambda,q, loop, remove_masked_batches):
        while True:
            gen = gen_lambda()
            while True:
                try:
                    (next, any_masked) = gen.__next__()
                    if remove_masked_batches and any_masked>0:
                        continue
                    q.put(next)
                except StopIteration:
                    break
            if not loop:
                #print(loop,"finished with worker")
                q.put(None)
                break
        return

    @staticmethod
    def _make_queue(gen_lambda, queue_max = 100, loop=False, remove_masked_batches = False):
        q = Queue(maxsize=queue_max)
        t = Thread(target=DataGenerator._worker, args=(gen_lambda, q, loop, remove_masked_batches))
        t.start()
        return (q.get,t)


    def get_batch(self, type, loop = None, remove_masked_batches=False):
        if loop is None:
            if type == "train":
                (queu_batch,t) = self._make_queue(self.generatorTrain, loop=False, remove_masked_batches=True)
            elif type == "val":
                (queu_batch, t) = self._make_queue(self.generatorVal, loop=False)
            elif type == "test":
                (queu_batch, t) = self._make_queue(self.generatorTest, loop=False)
        else:
            if type == "train":
                (queu_batch, t) = self._make_queue(self.generatorTrain, loop=loop, remove_masked_batches=remove_masked_batches)
            elif type == "val":
                (queu_batch, t) = self._make_queue(self.generatorVal, loop=loop, remove_masked_batches=remove_masked_batches)
            elif type == "test":
                (queu_batch, t) = self._make_queue(self.generatorTest, loop=loop, remove_masked_batches=remove_masked_batches)
        return queu_batch


    def get_sample_train(self):
        sample = self.train_generator()
        if sample is None:
            print("starting new training batch")
            self.train_generator = self.get_batch("train")
            sample = self.train_generator()
        return sample

    def empty_train(self):
        i = 0
        while 1:
            sample=self.train_generator()
            if sample is None:
                break
            i+=1
        return i



    def __init__(self,datasetname, folder_data, folder_result, batch_size, is_Q_A, val_size, max_len=np.infty, w2v="gloveModelBin.bin"):
        self.dataset_name = datasetname
        self.folder_data = folder_data
        self.folder_result = folder_result
        self.batch_size = batch_size
        self.is_Q_A = is_Q_A
        self.val_size = val_size
        self.max_len = max_len
        self.threads = []

        self.trans_dic, self.number_targets, self.vocab_size, self.generatorTrain, self.generatorVal, self.generatorTest = \
            DataGenerator.generate_data_from_name(self.dataset_name, self.folder_data, self.folder_result, self.batch_size, self.is_Q_A, self.val_size,
                                                  self.max_len)

        if w2v is not None:
            w2v = KeyedVectors.load_word2vec_format(
                folder_data + w2v, binary=True)
            self.w2v_for_data = DataGenerator.matrix_for_embed(w2v,self.trans_dic)
        else:
            self.w2v_for_data=None

        self.train_generator = self.get_batch("train")



def checkLengthDataSets(dataset_name, folder_data, folder_result, is_Q_A, testing_batch_size=100):
    batch_size = 1
    percentiles = [100, 95,90]
    dg = DataGenerator(dataset_name, folder_data, folder_result, batch_size, is_Q_A)
    sampler_train = dg.get_batch(type="train", loop=False)
    sampler_val = dg.get_batch(type="val", loop=False)

    lengths_single_batch = []
    return_dic = {}
    for sampler in [sampler_train, sampler_val]:
        while 1:
            sample = sampler()
            if sample is not None:
                sample = sample[0]
                lengths_single_batch.append(sample.shape[0])
            else:
                break
    #test percentiles:
    print("---------------",dataset_name,"---------------")
    for percentile in percentiles:
        print(str(percentile) + "%", np.percentile(lengths_single_batch, percentile))

    batch_size=testing_batch_size
    for percentile in percentiles:
        if percentile == 100:
            max_len = np.infty
        else:
            max_len = np.percentile(lengths_single_batch,percentile)
        dg = DataGenerator(dataset_name, folder_data, folder_result, batch_size, is_Q_A, max_len=max_len)
        sampler_train = dg.get_batch(type="train", loop=False)
        sampler_val = dg.get_batch(type="val", loop=False)
        lengths = []
        for sampler in [sampler_train, sampler_val]:
            while 1:
                sample = sampler()
                if sample is not None:
                    sample = sample[0]
                    lengths.append(sample.shape[0])
                else:
                    break
        print("percentile: ", percentile, np.mean(lengths))
        return_dic[(dataset_name,percentile)] = max_len
    print("------------------------------")
    return return_dic


def get_data_set_statistics(dataset_name, folder_data, folder_result, is_Q_A):
    dg = DataGenerator(dataset_name, folder_data, folder_result, 1, is_Q_A,val_size=0.15)
    sampler_train = dg.get_batch(type="train", loop=False)
    sampler_val = dg.get_batch(type="val", loop=False)
    test_val = dg.get_batch(type="test", loop=False)

    lengths = []
    train_i = 0
    val_i = 0
    test_i = 0
    classes = set()

    for sampler in [sampler_train]:
        while 1:
            sample  = sampler()
            if sample is not None:
                train_i += 1
                sample= sample[0]
                lengths.append(sample.shape[0])
            else:
                break

    for sampler in [sampler_val]:
        while 1:
            sample  = sampler()
            if sample is not None:
                val_i += 1
                sample= sample[0]
                lengths.append(sample.shape[0])
            else:
                break

    for sampler in [test_val]:
        while 1:
            sample  = sampler()
            if sample is not None:
                test_i += 1
                sample= sample[0]
                lengths.append(sample.shape[0])
            else:
                break

    print(dataset_name, np.mean(lengths),train_i,val_i,test_i,len(dg.trans_dic))





if __name__ == "__main__":
    folder_data = "data/"
    folder_result = "split_folder"
    #is_Q_A = False
    return_dics = []
    for (dataset_name, is_Q_A) in [("agnews",False)]:#[("dbpedia",False),("IMDB",False),("yelp",False),("CN_onetext",True),("NE_onetext",True)]: #[]:
        get_data_set_statistics(dataset_name,folder_data,folder_result, is_Q_A)

