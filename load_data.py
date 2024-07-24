# Code reused from https://github.com/jennyzhang0215/DKVMN.git
import os.path

import numpy as np
import math
import torch
from tqdm import tqdm
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

class DATA(object):
    def __init__(self, n_question, seqlen, separate_char, name="data"):
        # In the ASSISTments2009 dataset:
        # param: n_queation = 110
        #        seqlen = 200
        self.separate_char = separate_char
        self.n_question = n_question
        self.seqlen = seqlen
    # data format
    # id, true_student_id
    # 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    # 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0

    def load_data(self, path):
        f_data = open(path, 'r')
        q_data = []
        qa_data = []
        idx_data = []
        for lineID, line in tqdm( enumerate(f_data)):
            line = line.strip()
            # lineID starts from 0
            if lineID % 3 == 0:
                student_id = lineID//3
            if lineID % 3 == 1:
                Q = line.split(self.separate_char)
                if len(Q[len(Q)-1]) == 0:
                    Q = Q[:-1]
                # print(len(Q))
            elif lineID % 3 == 2:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]
                # print(len(A),A)

                # start split the data
                n_split = 1
                # print('len(Q):',len(Q))
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1
                # print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    answer_sequence = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(Q[i]) > 0:
                            # int(A[i]) is in {0,1}
                            Xindex = int(Q[i]) + int(A[i]) * self.n_question
                            question_sequence.append(int(Q[i]))
                            answer_sequence.append(Xindex)
                        else:
                            print(Q[i])
                    #print('instance:-->', len(instance),instance)
                    q_data.append(question_sequence)
                    qa_data.append(answer_sequence)
                    idx_data.append(student_id)
        f_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat
        # dataArray: [ array([[],[],..])] Shape: (3633, 200)
        return q_dataArray, qa_dataArray, np.asarray(idx_data)


class PID_DATA(object):
    def __init__(self, n_question,  seqlen, separate_char, name="data"):
        # In the ASSISTments2009 dataset:
        # param: n_queation = 110
        #        seqlen = 200
        self.separate_char = separate_char
        self.seqlen = seqlen
        self.n_question = n_question
        question_dict = {}

        # 读取文件
        if self.n_question==301:
            with open('./dataset/Ednet/question_dict.txt', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        question_id, tags_str = line.split(':')
                        tags_list = tags_str.split(';')
                        question_dict[question_id.strip()] = tags_list
        else:
            with open('./dataset/Assist2009/question_dict.txt', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        question_id, tags_str = line.split(':')
                        tags_list = tags_str.split(',')
                        question_dict[question_id.strip()] = tags_list
        self.question_tags=question_dict
    # data format
    # id, true_student_id
    # pid1, pid2, ...
    # 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    # 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0

    def load_data(self, path,dataname):
        if not (os.path.exists(dataname+'_tagsList.npy')):

            f_data = open(path, 'r')
            q_data = []
            a_data = []
            p_data = []
            for lineID, line in tqdm(enumerate(f_data)):
                line = line.strip()
                # lineID starts from 0
                if lineID % 3 == 0:
                    #第一行id
                    student_id = lineID//4
                # if lineID % 4 == 2:
                #     #第三行skill
                #     Q = line.split(self.separate_char)
                #     if len(Q[len(Q)-1]) == 0:
                #         Q = Q[:-1]
                # print(len(Q))
                if lineID % 3 == 1:
                    #第二行problem
                    P = line.split(self.separate_char)
                    if len(P[len(P) - 1]) == 0:
                        P = P[:-1]

                elif lineID % 3 == 2:
                    #第四行answer
                    A = line.split(self.separate_char)
                    if len(A[len(A)-1]) == 0:
                        A = A[:-1]
                    # print(len(A),A)

                    # start split the data
                    n_split = 1
                    # print('len(Q):',len(Q))
                    if len(P) > self.seqlen:
                        n_split = math.floor(len(P) / self.seqlen)
                        if len(P) % self.seqlen:
                            n_split = n_split + 1
                    # print('n_split:',n_split)
                    for k in range(n_split):
                        question_sequence = []
                        problem_sequence = []
                        answer_sequence = []
                        if k == n_split - 1:
                            endINdex = len(A)
                        else:
                            endINdex = (k+1) * self.seqlen
                        for i in range(k * self.seqlen, endINdex):
                            if len(P[i]) > 0:
                                problem_sequence.append(int(P[i]))
                                answer_sequence.append(int(A[i]))
                            else:
                                print(P[i])
                        q_data.append(question_sequence)
                        a_data.append(answer_sequence)
                        p_data.append(problem_sequence)

            f_data.close()
            ### data: [[],[],[],...] <-- set_max_seqlen is used
            # convert data into ndarrays for better speed during training
            q_dataArray = np.zeros((len(q_data), self.seqlen))
            for j in range(len(q_data)):
                dat = q_data[j]
                q_dataArray[j, :len(dat)] = dat

            a_dataArray = np.zeros((len(a_data), self.seqlen))-1
            for j in range(len(a_data)):
                dat = a_data[j]
                a_dataArray[j, :len(dat)] = dat

            p_dataArray = np.zeros((len(p_data), self.seqlen))
            tagsList=[]
            qaList=[]

            for j in tqdm( range(len(p_data))):
                dat = p_data[j]
                ans=a_data[j]
                tag=[]
                qa_data=[]
                p_dataArray[j, :len(dat)] = dat
                for t in range(len(dat)):
                    ttt=np.array(list(map(int,self.question_tags[str(dat[t])])))
                    tag_a=ttt+self.n_question*ans[t]
                    t_zero=np.zeros((7))
                    t_zero[:len(ttt)]=ttt
                    ta_zero=np.zeros((7))
                    ta_zero[:len(tag_a)]=tag_a
                    tag.append((t_zero))
                    qa_data.append(ta_zero)
                for zer in range(len(tag),self.seqlen):
                    tag.append(np.array([0,0,0,0,0,0,0]))
                    qa_data.append(np.array([0,0,0,0,0,0,0]))


                tag=np.array(tag)
                tagsList.append(tag)

                qa_data=np.array(qa_data)
                qaList.append(qa_data)




            tagsList=np.array(tagsList)
            qaList=np.array(qaList)

            np.save(dataname+'_tagsList.npy',tagsList)
            np.save(dataname+'qaList.npy',qaList)
            np.save(dataname+'p_dataArray.npy',p_dataArray)
            np.save(dataname+'a_dataArray.npy',a_dataArray)

        else:
            tagsList=np.load(dataname+'_tagsList.npy')
            qaList=np.load(dataname+'qaList.npy')
            p_dataArray=np.load(dataname+'p_dataArray.npy')
            a_dataArray=np.load(dataname+'a_dataArray.npy')
        return tagsList, qaList, p_dataArray,a_dataArray
