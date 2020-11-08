import random


class DataGenerate:

    def __init__(self, source,
                 batch_size=128,
                 maxlen=100,
                 train_flag=0
                 ):
        self.read(source)
        self.users = list(self.users)

        self.batch_size = batch_size
        self.eval_batch_size = batch_size
        self.train_flag = train_flag
        self.maxlen = maxlen
        self.index = 0

    # source: userid, itemid, sql_num
    def read(self, source):
        self.graph = {}
        self.users = set()
        self.items = set()
        with open(source, 'r') as f:
            for line in f:
                # [userid, itemid, sql_num]
                conts = line.strip().split(',')
                user_id = int(conts[0])
                item_id = int(conts[1])
                time_stamp = int(conts[2])
                self.users.add(user_id)
                self.items.add(item_id)
                if user_id not in self.graph:
                    self.graph[user_id] = []
                # {userid: (itemid, sql_num)}
                self.graph[user_id].append((item_id, time_stamp))
        for user_id, value in self.graph.items():
            # 根据时间戳排序
            value.sort(key=lambda x: x[1])
            self.graph[user_id] = [x[0] for x in value]
        self.users = list(self.users)
        self.items = list(self.items)

    def outData(self):
        user_id_list = self.users
        item_id_list = []
        hist_item_list = []
        hist_mask_list = []
        for user_id in user_id_list:
            item_list = self.graph[user_id]
            # 训练train
            if self.train_flag == 0:
                k = random.choice(range(4, len(item_list)))
                item_id_list.append(item_list[k])
            # 验证 valid
            else:
                k = int(len(item_list) * 0.8)
                item_id_list.append(item_list[k:])
            if k >= self.maxlen:
                hist_item_list.append(item_list[k - self.maxlen: k])
                hist_mask_list.append([1.0] * self.maxlen)
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.maxlen - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.maxlen - k))

        return (user_id_list, item_id_list), (hist_item_list, hist_mask_list)


def prepare_data(src, target):
    nick_id, item_id = src
    hist_item, hist_mask = target
    return nick_id, item_id, hist_item, hist_mask

train_file = '../data/ml-1m_data/ml-1m_train.txt'
# (user_id_list, item_id_list), (hist_item_list, hist_mask_list)
train_data = DataGenerate(train_file, 128, 20, train_flag=0).outData()
# print('train_data: ', len(train_data))

src, tgt = train_data
data_iter = prepare_data(src, tgt)
print(len(data_iter[0]))
print(len(data_iter[1]))
print(len(data_iter[2]))
print(len(data_iter[3]))
