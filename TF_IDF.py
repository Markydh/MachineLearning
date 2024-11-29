from collections import defaultdict
import math


def load_datasets():
    dataset = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    return dataset



def feature_select(list_words):
    # 计算每个单词的TF
    # 单词出现的次数 / 所有单词总个数
    word_dict = defaultdict(int)
    print(word_dict)
    for list_word in list_words:
        for i in list_word:
            word_dict[i] = word_dict[i]+1

    # 记录每个单词的tf值
    word_tf = defaultdict(int)
    for i in word_dict:
        word_tf[i] = word_dict[i] / sum(word_dict.values())

    print('每个单词的tf值\n', word_tf, len(word_tf))

    word_idf = defaultdict(int)
    # 记录每个单词的IDF
    for i in word_dict:
        count = 0
        for list_word in list_words:
            if i in list_word:
                count+=1
        word_idf[i] = math.log(len(list_words)/(count+1))

    print('每个单词的idf值\n', word_idf, len(word_idf))


    word_tf_idf = {}
    # 计算每个单词的 TF-IDF 值
    for i in word_tf:
        word_tf_idf[i] = word_tf[i] * word_idf[i]

    print('每个单词的tf-idf值\n', word_tf_idf)

if __name__ == "__main__":
    feature_select(load_datasets())