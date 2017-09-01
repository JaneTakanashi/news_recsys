# encoding=utf-8
"""split TrainData.txt into two parts.
    item_raw.csv:
        the csv header: ['itemId', 'title', 'content', 'pubtime']
    user_raw.csv:
        the csv header: ['userId', 'itemId',  'timestamp']
"""
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf8')

itemFileHeader = ['itemId', 'title', 'content', 'pubtime']
userFileHeader = ['userId', 'itemId',  'timestamp']

itemCSVFile = open('./data/raw_data/item_raw.csv','w')
userCSVFile = open('./data/raw_data/user_raw.csv','w')
itemWriter = csv.writer(itemCSVFile)
userWriter = csv.writer(userCSVFile)
itemWriter.writerow(itemFileHeader)
userWriter.writerow(userFileHeader)

id_dict = {}
with open('./data/raw_data/TrainData.txt', 'r') as f:
    for line in f.readlines():
        #TrainData.txt was splited by '\t'
        list = line.split('\t')

        item = []
        user = []

        # list[0]: 用户编号
        # list[1]: 新闻编号
        # list[2]: 浏览时间
        # list[3]: 新闻标题
        # list[4]: 新闻详细内容
        # list[5]:新闻发表时间
        user.append(list[0])
        user.append(list[1])
        user.append(list[2])
        userWriter.writerow(user)

        # 新闻去重
        if(id_dict.has_key(list[1])):
            continue
        else:
            id_dict[list[1]] = 1
            item.append(list[1])
            item.append(list[3])
            item.append(list[4])
            item.append(list[5])
            itemWriter.writerow(item)

itemCSVFile.close()
userCSVFile.close()
print('finished')
