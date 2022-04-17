from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import json
import numpy as np

count_article = 0
file_name = 'สุชยา เกษจำรัส'
for i in range(1,1009):
    with open(file_name+'\\'+file_name+str(i)+'.json',encoding="utf-8") as f:
        s = f.read()
    manage = json.loads(s)
    # print(manage)
    if(len(manage['text'])!=0):
        cvec = CountVectorizer(analyzer=lambda x:x.split(','))
        cv_fit = cvec.fit_transform(manage['text'])
        # print(cv_fit)
        word_list = cvec.get_feature_names()


        count_list = np.asarray(cv_fit.sum(axis=0))[0]
        a = dict(zip(word_list, count_list))
        try:
            print('บทความที่',str(i),'เจอมีทั้งหมด',a['ๆ'],'คำ')
            count_article += 1
            print('total',count_article)

        except:
            print('บทความที่',str(i),'ไม่เจอ')
    



