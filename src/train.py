#coding:utf-8
import argparse
import math
import random
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
import os
import faiss
from utils import *
from model import *
from tensorboardX import SummaryWriter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='train', help='train | test')
parser.add_argument('--dataset', type=str, default='ml-1m', help='ml-1m | ml-10m | book | taobao')
parser.add_argument('--random_seed', type=int, default=19)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--num_interest', type=int, default=4)
parser.add_argument('--model_type', type=str, default='none', help='DNN | GRU4REC | ..')
parser.add_argument('--learning_rate', type=float, default=0.001, help='')
parser.add_argument('--max_iter', type=int, default=1000, help='(k)')
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--coef', default=None)
parser.add_argument('--topN', type=int, default=50)
parser.add_argument('--test_iter', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--maxlen', type=int, default=50)
parser.add_argument('--num_blocks', type=float, default=2)
parser.add_argument('--dropout_rate', type=float, default=0.2)

best_metric = 0

def prepare_data(src, target):
    nick_id, item_id = src
    hist_item, hist_mask = target
    return nick_id, item_id, hist_item, hist_mask

def load_item_cate(source):
    item_cate = {}
    with open(source, 'r') as f:
        for line in f:
            # 先移除字符串首尾的空格再按‘,’切分
            conts = line.strip().split(',')
            item_id = int(conts[0])
            cate_id = int(conts[1])
            item_cate[item_id] = cate_id
    return item_cate

def compute_diversity(item_list, item_cate_map):
    n = len(item_list)
    diversity = 0.0
    for i in range(n):
        for j in range(i+1, n):
            diversity += item_cate_map[item_list[i]] != item_cate_map[item_list[j]]
    diversity /= ((n-1) * n / 2)
    return diversity

def evaluate_full(test_data, model, model_path, batch_size, item_cate_map, save=True, coef=None):
    topN = args.topN

    item_embs = model.output_item()

    try:
        gpu_index = faiss.IndexFlatL2(args.embedding_dim)
        gpu_index.add(item_embs)
    except Exception as e:
        return {}

    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_diversity = 0.0

    src, tgt = test_data
    nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt)

    user_embs =  model.predict(hist_item)

    # 多个兴趣表示[num_heads, embedding_dim]
    if len(user_embs.shape) == 2:
        # I: itemList    D:distance
        # I, D = item_embs_index.get_nns_by_vector(user_embs, topN, include_distances=True)
        D, I = gpu_index.search(user_embs, topN)
        for i, iid_list in enumerate(item_id):
            recall = 0
            dcg = 0.0
            item_list = set(I[i])
            for no, iid in enumerate(iid_list):
                if iid in item_list:
                    recall += 1
                    dcg += 1.0 / math.log(no+2, 2)
            idcg = 0.0
            for no in range(recall):
                idcg += 1.0 / math.log(no+2, 2)
            total_recall += recall * 1.0 / len(iid_list)
            if recall > 0:
                total_ndcg += dcg / idcg
                total_hitrate += 1
            if not save:
                total_diversity += compute_diversity(I[i], item_cate_map)
    else:
        ni = user_embs.shape[1]
        user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]])
        # I, D = item_embs_index.get_nns_by_vector(user_embs, topN, include_distances=True)
        D, I = gpu_index.search(user_embs, topN)
        for i, iid_list in enumerate(item_id):
            recall = 0
            dcg = 0.0
            item_list_set = set()
            if coef is None:
                item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                item_list.sort(key=lambda x:x[1], reverse=True)
                for j in range(len(item_list)):
                    if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                        item_list_set.add(item_list[j][0])
                        if len(item_list_set) >= topN:
                            break
            else:
                origin_item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                origin_item_list.sort(key=lambda x:x[1], reverse=True)
                item_list = []
                tmp_item_set = set()
                for (x, y) in origin_item_list:
                    if x not in tmp_item_set and x in item_cate_map:
                        item_list.append((x, y, item_cate_map[x]))
                        tmp_item_set.add(x)
                cate_dict = defaultdict(int)
                for j in range(topN):
                    max_index = 0
                    max_score = item_list[0][1] - coef * cate_dict[item_list[0][2]]
                    for k in range(1, len(item_list)):
                        if item_list[k][1] - coef * cate_dict[item_list[k][2]] > max_score:
                            max_index = k
                            max_score = item_list[k][1] - coef * cate_dict[item_list[k][2]]
                        elif item_list[k][1] < max_score:
                            break
                    item_list_set.add(item_list[max_index][0])
                    cate_dict[item_list[max_index][2]] += 1
                    item_list.pop(max_index)

            for no, iid in enumerate(iid_list):
                if iid in item_list_set:
                    recall += 1
                    dcg += 1.0 / math.log(no+2, 2)
            idcg = 0.0
            for no in range(recall):
                idcg += 1.0 / math.log(no+2, 2)
            total_recall += recall * 1.0 / len(iid_list)
            if recall > 0:
                total_ndcg += dcg / idcg
                total_hitrate += 1
            if not save:
                total_diversity += compute_diversity(list(item_list_set), item_cate_map)

    total += len(item_id)

    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total
    diversity = total_diversity * 1.0 / total

    if save:
        return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'diversity': diversity}

def save(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    model.save(path + 'model.h5')

def get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=True):
    extr_name = input('Please input the experiment name: ')
    para_name = '_'.join([dataset, model_type, 'b'+str(batch_size), 'lr'+str(lr), 'd'+str(args.embedding_dim), 'len'+str(maxlen)])
    exp_name = para_name + '_' + extr_name

    while os.path.exists('runs/' + exp_name) and save:
        flag = input('The exp name already exists. Do you want to cover? (y/n)')
        if flag == 'y' or flag == 'Y':
            shutil.rmtree('runs/' + exp_name)
            break
        else:
            extr_name = input('Please input the experiment name: ')
            exp_name = para_name + '_' + extr_name

    return exp_name

def train(
        train_file,
        valid_file,
        test_file,
        cate_file,
        item_count,
        dataset = "book",
        batch_size = 128,
        maxlen = 100,
        test_iter = 50,
        model_type = 'DNN',
        lr = 0.001,
        max_iter = 100,
        patience = 20
):
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen)

    best_model_path = "best_model/" + exp_name + '/'

    writer = SummaryWriter('runs/' + exp_name)


    item_cate_map = load_item_cate(cate_file)

    # (user_id_list, item_id_list), (hist_item_list, hist_mask_list)
    train_data = DataGenerate(train_file, batch_size, maxlen, train_flag=0).outData()
    valid_data = DataGenerate(valid_file, batch_size, maxlen, train_flag=1).outData()

    model = MSARec(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest,
                   args.num_interest, maxlen=args.maxlen, dropout_rate=args.dropout_rate, num_blocks=2)
    # print(model.summary())
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, decay=0.1)

    model.compile(loss=model.sampled_softmax_loss,
                  optimizer=optimizer, metrics=[])

    print('training begin')
    sys.stdout.flush()

    callbacks = [
        # 如果`val_loss`在2个以上的周期内停止改进，则进行中断训练
        tf.keras.callbacks.EarlyStopping(patience=args.patience, monitor='val_loss'),
        # 将TensorBoard日志写入`./logs`目录
        tf.keras.callbacks.TensorBoard(log_dir='logs')
    ]

    start_time = time.time()
    try:
        trials = 0
        src, tgt = train_data
        src2, tgt2 = valid_data
        nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt)
        nick_id2, item_id2, hist_item2, hist_mask2 = prepare_data(src2, tgt2)
        # train_data: userid, itemid, sql_num
        for iter in range(max_iter):
            print('this is', iter, 'iter')
            model.fit(
                hist_item,
                item_id,
                epochs=1,
                callbacks=callbacks,
                validation_data=(hist_item2, item_id2),
                batch_size=batch_size,
            )

            if iter % test_iter == 0:
                metrics = evaluate_full(valid_data, model, best_model_path, batch_size, item_cate_map)
                log_str = ''
                if metrics != {}:
                    log_str += ', ' + ', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()])
                print(exp_name)
                print(log_str)

                if metrics != {}:
                    for key, value in metrics.items():
                        writer.add_scalar('eval/' + key, value, iter)

                if 'recall' in metrics:
                    recall = metrics['recall']
                    global best_metric
                    if recall > best_metric:
                        best_metric = recall
                        save(model, best_model_path)
                        trials = 0
                    else:
                        trials += 1
                        if trials > patience:
                            break

                test_time = time.time()
                print("time interval: %.4f min" % ((test_time-start_time)/60.0))
                sys.stdout.flush()

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    save(model, best_model_path)

    metrics = evaluate_full(valid_data, model, best_model_path, batch_size, item_cate_map, save=False)
    print(', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()]))

    test_data = DataGenerate(test_file, batch_size, maxlen, train_flag=2).outData()
    metrics = evaluate_full(test_data, model, best_model_path, batch_size, item_cate_map, save=False)
    print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))

if __name__ == '__main__':
    print(sys.argv)
    args = parser.parse_args()
    SEED = args.random_seed

    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_name = 'train'
    valid_name = 'valid'
    test_name = 'test'

    if args.dataset == 'ml-1m':
        path = '/content/MIKRec-2.0/data/ml-1m_data/'
        # path = '../data/ml-1m_data/'
        item_count = 3417
        batch_size = args.batch_size
        maxlen = args.maxlen
        test_iter = args.test_iter
    elif args.dataset == 'book':
        path = '/content/MIKRec-2.0/data/book_data/'
        item_count = 367983
        batch_size = 128
        maxlen = 20
        test_iter = 1000
    elif args.dataset == 'ml-10m':
        path = '/content/MIKRec-2.0/data/ml-10m_data/'
        item_count = 10197
        batch_size = args.batch_size
        maxlen = args.maxlen
        test_iter = args.test_iter

    train_file = path + args.dataset + '_train.txt'
    valid_file = path + args.dataset + '_valid.txt'
    test_file = path + args.dataset + '_test.txt'
    cate_file = path + args.dataset + '_item_cate.txt'
    dataset = args.dataset

    if args.p == 'train':
        train(train_file=train_file, valid_file=valid_file, test_file=test_file, cate_file=cate_file,
              item_count=item_count, dataset=dataset, batch_size=batch_size, maxlen=maxlen, test_iter=test_iter,
              model_type=args.model_type, lr=args.learning_rate, max_iter=args.max_iter, patience=args.patience)
    else:
        print('do nothing...')
