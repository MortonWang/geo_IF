'''
@File  :doc2vec.py
@Author:Morton
@Date  :2020/6/18  19:48
@Desc  : get content features by using dec2vec model.
'''
# -*- coding:utf-8 -*-
import smart_open
import gensim


def preprocess(raw_file, process_file):
    # Import and download stopwords from NLTK.
    from nltk.corpus import stopwords
    from nltk import download
    download('stopwords')  # Download stopwords list.
    stop_words = stopwords.words('english')
    interpunction = "||| 《 》 # @ ' —— + - . ! ！ / _ , $ ￥ % ^ * ( ) ] | [ ， 。 ？ ? : ： ； ; 、 ~ …… … & * （ ）".split()

    with open(raw_file, 'r') as f_in:
        lines = f_in.readlines()
    f_in.close()
    total_line = len(lines)
    print("{} total line is:{}".format(raw_file, total_line))

    str_out = " "
    for index, line in enumerate(lines):
        line = line.lower().split()
        line = [w for w in line if w not in interpunction]      # Remove interpunction.
        line = [w for w in line if w not in stop_words]         # Remove stopwords.
        line = ' '.join(line) + "\n"
        str_out = str_out + line
        if (index + 1) % 1000 == 0 or index == (total_line - 1):
            with open(process_file, 'a') as f_out:
                f_out.write(str_out)
            print("Process num {} , and average of word is {}".format(index + 1, len(str_out.split()) / 1000))
            str_out = " "
    f_out.close()

    # verify the line num of files.
    with open(process_file, 'r') as f_in:
        lines_fin = f_in.readlines()
        print("{} total line is:{}".format(process_file, len(lines_fin)))
    f_in.close()


def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


def gensim_models_doc2vec(raw_file="./content_all.txt", doc2vec_model_file="./model_dim_512_epoch_40.bin"):
    process_file = "../dataset_cmu/corpus/content_all_process.txt"
    preprocess(raw_file, process_file)
    all_corpus = list(read_corpus(process_file))

    # get model
    model = gensim.models.doc2vec.Doc2Vec(vector_size=512, min_count=2, epochs=40, window=10, sample=1e-3,
                                          negative=5, workers=3, hs=0, ns_exponent=0.75, dm=0)
    print("start build ...")
    model.build_vocab(documents=all_corpus, progress_per=10000, keep_raw_vocab=False)
    print("start train ...")
    model.train(documents=all_corpus, total_examples=model.corpus_count, epochs=model.epochs, word_count=0)
    model.init_sims(replace=True)

    # save model
    model.save(doc2vec_model_file)
    print("doc2vec save success into: {}".format(doc2vec_model_file))
