import tools
import jieba
import re
from gensim import corpora
from gensim.models import LdaModel

stopped_words = tools.get_stopped_words()
trainset = []
fr = open('corpus.txt', 'r', encoding='utf-8')
fw = open('corpus_word.txt', 'w', encoding='utf-8')
for line in fr:
    line = jieba.cut(line.rstrip(), cut_all=False)
    words = [word for word in line if word not in stopped_words]
    trainset.append(words)
    fw.writelines(" ".join(words) + "\n")
fr.close()
fw.close()
# 制作一个字典
dictionary = corpora.Dictionary(trainset)
# 训练集中每个词语的个数元组
corpus = [dictionary.doc2bow(text) for text in trainset]
lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=4)
# lda.save('*.model')
result = lda.print_topics(4)
finalwords = []
for tuple in result:
    _, string = tuple
    words = re.findall(r"\"(.+?)\"", string)
    for word in words:
        if word not in finalwords:
            finalwords.append(word)
tools.list_to_file("./LDA_topic_word.txt", finalwords)
