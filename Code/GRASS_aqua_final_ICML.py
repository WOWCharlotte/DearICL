from pyexpat.errors import XML_ERROR_XML_DECL
import numpy as np
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import torch
from statistics import mean
from random import sample, randint
import re
import pickle 
import json
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential
import os
from ollama import chat, ChatResponse
from ltr import train_ltr_model, MSLRDataset, DataLoader, pirank_simple_loss, RankNet
from torch.optim import lr_scheduler
from ast import literal_eval
import bert_score 
from itertools import zip_longest
import ast
from collections import defaultdict

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scorer = bert_score.BERTScorer(lang='en')



#####################################################################################################
system_message = """
Follow given examples, Think step by step and Solve the math question and give the concise short rationale as reasoning steps and not long ones strictly preceded by A: and then generate only the correct option letter based on reasoning steps preceded by Final Answer:
####

"""
api_keys = ["ollama"]
endpoint_urls = ["http://localhost:11434/v1/"]
llm_names = []

# Gen response from API
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def get_completion(prompt, api_key, endpoint_url, hard_code_exception=False):

    response: ChatResponse = chat(model='llama3.2:3b',  messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                        ],
        keep_alive="24h",
        options = {'temperature':0.25,
        'top_k':10,
        'top_p':1.0})
    return response["message"]["content"].strip()

###############################################################################################



def llm_output(user_query, hard_code_exception=False):
    results = get_completion(user_query, api_keys[0], endpoint_urls[0], hard_code_exception=hard_code_exception)
    #results = get_completion(user_query)
    
    return results


# Self consistency on 10 generated answers
def self_con(tmp_list):
    ans_list = []
    for tmp in tmp_list:
        print("tmp",tmp)
        ans = ""
        if len(tmp.split("Final Answer:"))>0:
            ans = tmp.split("Final Answer:")[-1]
            ans = ans.split("\n")[0]
            # print(ans)
            if "each" in ans:  ans = ans.split("each")[0]
            if "=" in ans: ans = ans.split("=")[-1]
            ans = re.sub(r'[^0-9.]',"",ans)
            if len(ans)>0 and ans[-1]==".": ans = ans[:-1]
            # print(ans, "**************")
            try:
                float(ans)
                ans = round(float(ans))
                ans_list.append(ans)
            except: pass
        # ans_list.append(ans)

    # print(ans_list)
    d = {}
    for i in ans_list:
        if i=="":
            continue
        if int(i) in d:
            d[int(i)] += 1
        else:
            d[int(i)] = 1
    # print(d)
    n = sorted(d.items(), key=lambda x:x[1], reverse=True)
    return n

# Strip answer from sentence
def clean_ans(s):
    ans_s = s.split("#### ")[1]
    ans_s = ans_s.replace(",","")
    return ans_s

def get_prompt(ex):
    s = "\n\n"
    s += "Question:" + ex["question"]+"\n"
    ex["answer"] = re.sub("<<.*?>>", "", ex["answer"])
    ex["answer"] = ex["answer"].replace("#### ", "Final Answer:")
    s += ex["answer"]
    return s


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def extract_option(output: str) -> str:
    matches = re.findall(r'([A-E]\) |[A-E])', output) #'([A-E]\))'
    return matches[-1].strip(')')[0] if matches else None

def prompt_for_manual_prediction(ex, shots):

    stop_signal = "\n\n"
    showcase_examples = [
            "Q: {}\nO: {} \nA: {}. Final Answer: {}\n".format(
                 s["question"],s["options"],
                s["rationale"], s["correct"]) for index, s in shots.iterrows()
        ]

    #print("showcase_examples",showcase_examples[0])

    prompt = "\n".join(showcase_examples)
    # prompt=prompt+"\n\n{text}\n"


    input_example = prompt + " \n Following the above examples, solve the Test question: \nQ: {}\n O:{} by thinking step by step and strictly  give rationale strictly preceded by A: and then strictly only output the corresponding correct option letter strictly preceded by Final Answer:".format(ex['question'], ex['options'])

    return prompt
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def LLM_avg_error(exemplars_set, val_data):
    # stop_signal = "\n\n"
    error=[]
    # increment error if model predicted answer is not equal to the ground truth answer for the test question by passing the exemplars
    for exemplars in tqdm(exemplars_set,total=len(exemplars_set),desc="LLM Loss Fn Exemplar"):
        matches = 0
        mismatches = 0
        exnum = 0
        # acc_records = []
        for index, row in val_data.iterrows():
            print("********************************+")
            prompt = prompt_for_manual_prediction(row, exemplars)
            tmp_list = llm_output(prompt)
            #print(tmp[0])
            
            answer = extract_option(tmp_list)
            #if answer=="" and len(n)>1: answer = n[1][0]

            # ans = ""
            # if len(tmp_list[0].split("The option is "))>6:
            #     ans = tmp_list[0].split("The option is ")[6][0]
            # answer=ans
            
            print("\nAnswer: ", answer)
            gt = row["correct"]
            print("GT: ", gt)
            if(answer==gt):
              matches+=1
            else:
              mismatches+=1

        error.append(mismatches/(matches+mismatches))


    return error
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def LLM_error_indicator(exemplars, val_data, val_emb = None):
    # stop_signal = "\n\n"
    #arm_rewards=[]
    # increment error if model predicted answer is not equal to the ground truth answer for the test question by passing the exemplars
    subset_accuracy = 0
    gt = []
    answers = []
    for index, row in val_data.iterrows():
        prompt = prompt_for_manual_prediction(row, exemplars)
        tmp_list = llm_output(prompt)
        #n = self_con(tmp_list)
        #answer = n[0][0]
        
        rationale = row["rationale"]

        answers.append(tmp_list)
        gt.append(rationale)
        #if answer==ground_truth: subset_accuracy = 1
        #else: subset_accuracy = 0

    P, R, F1 = scorer.score(answers, gt)
    #print("subset_accuracy",subset_accuracy)
    scores = np.round((F1*10).numpy()) #convert to pytorch
    averaged_val_acc = scores.sum()/val_data.shape[0]
    return averaged_val_acc, scores.tolist()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def convert_dict(d, key_order=None):
    """
    Convert a dictionary of lists into a list of tuples.

    Parameters
    ----------
    d : dict
        Keys map to lists; all lists should be the same length.
    key_order : list, optional
        If provided, determines the order of the tuple items by specifying
        which keys (and in what order) to pull from `d`. Otherwise uses
        the dict’s insertion order.

    Returns
    -------
    list of tuple
        One tuple per index, with elements taken from each list in `d`.
    """
    if key_order is not None:
        lists = [d[k] for k in key_order]
    else:
        lists = d.values()
    return list(zip(*lists))

def generate_test_input(i,L_ind,train_emb,test_emb):
    data = {'q_id':[], 'features': [], 'score':[]}
    for k in range(len(L_ind)):
        indexes = L_ind[k]
        bools = [False if x not in indexes else True for x in range(train_emb.shape[0])]
        features = train_emb[bools]
        mean = np.mean(np.vstack((features,np.expand_dims(test_emb[i], axis = 0))), axis = 0)
        data['q_id'].append(int(i))
        data['features'].append(torch.from_numpy(mean))
    
    dummy_scores = [randint(0,1) for x in range(len(data['q_id']))]
    data['score'] = dummy_scores
    test_input = convert_dict(data)
    return test_input

class top_m_arm:

    def __init__(self, args, X, m, model,optimizer,learning_rate=0.0001, weight_decay=1e-4,batch_size = 16, num_features = 768,list_size = 100, val_data=None, subsets = None, theta=None, delta=0.05, epsilon=0.1):
        self.not_terminated = 0
        self.current_n_simu = 0
        self.delta = delta
        self.epsilon = epsilon
        self.m = m
        self.val_data=val_data
        self.subsets = subsets
        self.X = X
        self.N, self.K = np.shape(X)
        self.arms = list(range(self.K))
        self.true_theta = theta
        self.theta = None
        self.T_init = 0
        self.n_das = 5
        self.sigma=0.5
        self.index_time = 0
        self.model = model
        self.optimizer = optimizer#torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        self.batch_size = batch_size
        self.list_size = list_size
        self.num_features = num_features
        


    def reset_bandit_specific_parameters(self):
        assert np.shape(self.X)[0] == self.N and np.shape(self.X)[1] == self.K and len(np.shape(self.X)) == 2
        self.T_init = 0
        self.hyper = 1.
        self.B_inv = 1/float(self.hyper**2)*np.eye(self.N)
        self.b = np.zeros(self.N)
        self.theta = np.random.normal(0, 1, size=(1, self.N)) 


    def clear(self):
        ## Collected rewards 
        self.rewards = [] 
        ## Pulled arms
        self.pulled_arms = []
        ## Should the learning step stop?
        self.done = False
        self.t = 0
        ## Estimated empirical average reward for each arm
        self.means = np.zeros(self.K)
        self.cum_sum = [0]*self.K 
        ## Contextual algorithms 
        self.B_inv = None 
        self.b = None 
        self.theta = None 
        ## Number of times each arm has been sampled so far 
        self.na = np.zeros(self.K) 
        ## Plot arguments 
        ## Confidence intervals (if needed) 
        self.confidence_intervals = None 
        self.best_arm, self.challenger = [None]*2 
        ## Stopping quantity 
        self.B = float("inf") 
        self.indices = None 
        ## "optimal" allocation 
        self.ratio = {}  
        self.reset_bandit_specific_parameters()


    def matrix_norm(self, x, A):
        '''Mahalanobis norm/Matrix norm: sqrt(x^TAx)'''
        #print("np.shape(x))",np.shape(x),np.shape(A))
        # if(len(np.shape(x))!=2):
        #     x = x.reshape((-1, 1))
        # print("np.shape(x))",np.shape(x),np.shape(A))
        #assert (is_of_type(x, "numpy.ndarray") or is_of_type(x, "numpy.matrix")) and is_of_type(A, "numpy.matrix") or is_of_type(A, "numpy.ndarray")
        assert len(np.shape(x)) == 2 and len(np.shape(A)) == 2
        assert np.shape(x)[0] == np.shape(A)[0] and np.shape(A)[0] == np.shape(A)[1]
        return np.sqrt((x.T).dot(A).dot(x))


    def gap(self, i, args, j=None):
        if self.t == 0:
            self.model.eval()
            assert all([0 <= x and x < np.shape(self.X)[1] for x in [i]])
            mean_i = []
            mean_j = []
            data_i = [self.pred_data[k+i] for k in range(0,len(self.pred_data),self.list_size)] #get validation data for arm i instead reproducing them each time
            data_j = [self.pred_data[k+j] for k in range(0,len(self.pred_data),self.list_size)]
            with torch.no_grad():
                for (q_id, features,score) in data_i:
                    mean_i.append(self.model(features.to(device)).item())
                for (q_id, features,score) in data_j:
                    mean_j.append(self.model(features.to(device)).item())
            if (str(j) != "None"):
                assert all([0 <= x and x < np.shape(self.X)[1] for x in [j]])
                gap_ = np.mean(np.array(mean_i)) - np.mean(np.array(mean_j))
            else:
                gap_ = np.mean(np.array(mean_i))
            return gap_ 
        
        if (str(j) != "None"):
            assert all([0 <= x and x < np.shape(self.X)[1] for x in [j]])
            gap_ = self.means[i] - self.means[j]
        else:
            gap_ = self.means[i]
        return gap_

    def HeuristicBeta(self,delta):
        #assert is_of_type(delta, "float")
        assert 0 < delta and delta < 1
        return (lambda t : np.log((np.log(t)+1)/float(delta)))

    def variance(self, i, t, args= None, j=None, c=1.96, epsilon_stab_t=0.01): 
          
        self.model.train()
        score_i = []
        score_j = []
        biases_1 = []
        biases_2 = []
        
        data_i = [self.pred_data[k+i] for k in range(0,len(self.pred_data),self.list_size)] #get validation data for arm i instead reproducing them each time
        data_j = [self.pred_data[k+j] for k in range(0,len(self.pred_data),self.list_size)]
        with torch.no_grad():
            for (q_id, features,score) in data_i:
                score_i.append(self.model(features.to(device)).item())
                #print("score_i[-1",score_i[-1], self.means[self.challenger])
                #biases_1.append(np.abs(score_i[-1]-self.means[self.challenger]))
            for (q_id, features,score) in data_j:
                score_j.append(self.model(features.to(device)).item())
                #biases_2.append(np.abs(score_j[-1]-np.max(self.means[self.J])))
        """
        data_i = self.pred_data[i]
        data_j = self.pred_data[j]
        with torch.no_grad():
            for i in range(20):
                score_i.append(self.model(data_i[1].to(device)).item())
                score_j.append(self.model(data_j[1].to(device)).item())
        """
        diffs = np.array(score_i) - np.array(score_j)
        V_hat = np.std(diffs)
        N = len(data_i)
        M = max(score_i/np.linalg.norm(score_i))
        log_term = np.log(2  / self.delta)
        c_new = c / np.sqrt(t) if t != 0 else c 

        mc_width = c_new * np.sqrt(2 * V_hat * log_term / N) + (2 * M * log_term) / (3 * N)
        epsilon_stab_t_new = epsilon_stab_t / np.sqrt(t) if t != 0 else epsilon_stab_t 
       # print("self.means[self.challenger]", self.means[self.challenger],self.challenger,i)
        challenge = self.challenger if self.challenger!=None else i
        W_t_new = mc_width + epsilon_stab_t_new + min([np.abs(np.mean(score_j) - self.means[j]), np.abs(np.mean(score_i)- self.means[challenge])]) 
        print("mc_widthmc_widthmc_width",mc_width,epsilon_stab_t_new, np.abs(np.mean(score_j) - self.means[j]), np.abs(np.mean(score_i)- self.means[challenge]))

        return W_t_new


    def index(self, i, j, t, args):
        gap_, var_ = self.gap(i, args, j=j), self.variance(i, t, args, j=j)
        print(f"{gap_=}")
        print(f"{var_=}")
        return gap_+var_


    def calculate_B_ij(self, i, j, t,args=None):
        if (t != self.index_time):
            self.B_di = {}
            self.index_time = t
        b = self.B_di.setdefault((i,j), float(self.index(i,j,t, args=args)))
        return b



    def B_ij(self, i, j):
        return self.calculate_B_ij(i, j, 
                               self.t+int(self.T_init == 0), 
                               self.arg_index())
    
    def randf(self,x, f):
        #x_float = is_of_type_LIST(x, "float")
        #assert x_float or is_of_type_LIST(x, "int")
        t = (float if (type(x[0]==float)) else int)
        x = np.array(x, dtype=t)
        f_ = lambda z : t(f(z))
        #assert is_of_type(f_(x), "float" if (x_float) else "int")
        smp = sample(np.argwhere(x == f_(x)).flatten().tolist(), 1)[0]
        #assert is_of_type(smp, "int")
        return smp


    def m_maximal(self, x, m):
        '''Returns @m-sized list of maximizing elements in @x IN ORDER (ties broken randomly)'''
        #assert is_of_type_LIST(x, "float") or is_of_type_LIST(x, "int")
        #assert is_of_type(m, "int")
        K = len(x)
        assert m > 0 and m <= K
        ids = np.argsort(x).flatten().tolist()
        ids.reverse()
        if (m < K):
            ids = [None]*m
            for i in range(m):
                id_ = self.randf(x, np.max)
                ids[i] = id_
                x[id_] = -float("inf")
        return ids


    def m_max(self, values, m):
        '''Returns element in @values which is the @mth maximal element'''
        arg_values = np.argsort(np.array(values).flatten()).tolist()
        return values[arg_values[-m]]


    #' @param A Numpy matrix: inverse of matrix
    def iterative_inversion(self, A, x):
        return A - (np.dot(np.dot(A, x), np.dot(x.T, A)))/float(1+self.matrix_norm(x, A)**2)


    def arg_index(self):
        return {"Sigma": (self.sigma**2)*self.B_inv, "theta": self.theta}


    def compute_index(self, j):
        #index = float(np.max([self.B_ij(i, j) for i in self.notJ]))
        index = float(np.max([self.B_ij(i, j) for i in self.N_t]))
        return index


    def compute_nt(self):
        print("self.J",self.J, self.means)
        Jt_means = self.means[self.J]
        min_means = np.min(Jt_means)
        worst_index = sample(np.argwhere(self.means == min_means).flatten().tolist(), 1)[0]
        print("--- in compute nt ", worst_index)
        while worst_index not in self.J:
            worst_index = sample(np.argwhere(self.means == min_means).flatten().tolist(), 1)[0]
        assert worst_index in self.J
        return worst_index


    def select_arm(self,nt):
        Jt_means = self.means[self.J]
        bt_means = np.max(Jt_means)	
        st_means = self.means[self.challenger]
        ct_means = self.means[self.c_t]
        nt_means = self.means[nt]
        print("!!!!!!!!!!!!!!!!!!!!!!! challenger, bt_means, st_means", self.challenger, bt_means,self.means[self.best_arm], st_means)
        print("self.B_ij(self.challenger, self.best_arm)",self.B_ij(self.worst_arm,self.best_arm),self.B_ij(self.best_arm,self.worst_arm),"bt,st"+str(self.B_ij(self.best_arm,self.challenger)),"st,bt"+str(self.B_ij(self.challenger, self.best_arm)),"nt,challenger"+str(self.B_ij(self.worst_arm,self.challenger)),"challenger,nt"+str(self.B_ij(self.challenger,self.worst_arm)))
        #self.B_ij(nt,self.challenger)<=self.epsilon
        #self.B_ij(nt,self.challenger)<self.B_ij(self.challenger,self.best_arm)
        #(nt_means-st_means)<=self.epsilon
        if(ct_means > nt_means ):
            selected_arm = self.c_t
            assert selected_arm in self.N_t
            print("**********inside select_arm",self.N_t)
            if type(self.N_t)==list:
                self.N_t.remove(selected_arm)
            else:
                self.N_t = self.N_t.tolist()
                self.N_t.remove(selected_arm)
            print("notJ",self.notJ)
            #print("**********inside select_arm after removing",self.N_t)
            return selected_arm
        else:
            return None


    def compute_Jt(self):
        print("----------", self.t)
        if(self.t==0):
            print("Simulation", self.means)
            return self.m_maximal(self.means.tolist(), self.m)
        else:
            print("***************** TIME T =", self.t)
            nt = self.worst_arm
            selected_arm = self.select_arm(nt)
            print("-------- Before updating Jt", self.J)
            self.previous_J=self.J
            print("****** worst arm, selected arm", nt, selected_arm)
            if (selected_arm is not None):
                self.J.remove(nt)
                self.J.append(selected_arm)
            if type(self.N_t)==list:
                self.N_t.append(nt)
            else:
                self.N_t = self.N_t.tolist()
                self.N_t.append(nt)
            print("After removing and appending", self.J)
            J_means = self.means[self.J]
            inds = J_means.argsort()[::-1].tolist()
            print(inds)
            self.J = np.array(self.J)[inds].tolist()
            if set(self.J)==set(self.previous_J):
                self.patience+=1
            else:
                self.patience=0
            print("After sorting",self.J)
            #J_means.sort(reverse=True)
            return self.J


    def compute_ct(self):
        #random.seed(42)
        #return self.J[randf(self.means[self.J], np.max)]
        Nt_means = self.means[self.N_t]
        max_means = np.max(Nt_means)
        best_index = sample(np.argwhere(self.means == max_means).flatten().tolist(), 1)[0]
        print("--- in compute bt ", best_index)
        while best_index not in self.N_t:
            best_index = sample(np.argwhere(self.means == max_means).flatten().tolist(), 1)[0]
        assert best_index in self.N_t
        return best_index


    def compute_bt(self):
        indices_J = [self.compute_index(i) for i in self.J]
        b_t = self.J[self.randf(indices_J, np.max)]
        return b_t


    def pull_arm_greedy(self, A):
        #direction = self.X[:, self.best_arm]-self.X[:, self.challenger]
        #uncertainty = [float(self.matrix_norm(direction, self.iterative_inversion(A, self.X[:,i]))) for i in self.arms]
        self.model.train()
        preds = []
        with torch.no_grad():
            for (q_id, features,score) in self.pred_data:
                preds.append(self.model(features.to(device)).item())
        preds = [preds[i:i + self.list_size] for i in range(0, len(preds), self.list_size)]
        uncertainty = np.std(np.array(preds), axis = 0).tolist()
        print(f"{len(uncertainty)}")
        print("uncertainty",uncertainty)
        a = self.arms[self.randf(uncertainty, np.min)]
        print("arm a", a, self.best_arm, self.challenger)
        return [int(a)]


    def sampling_rule(self):
        self.is_greedy = True
        return (self.pull_arm_greedy if (self.is_greedy) else self.pull_arm_optimized)(self.B_inv)


    def sample(self):
        print("self.B_inv",self.B_inv.shape)
        self.J = self.compute_Jt()
        print("----- J, arms", self.J, self.arms)
        self.notJ = [a for a in self.arms if (a not in self.J)]
        if(self.t== 0):
            self.N_t = np.random.choice(np.array(self.notJ), self.n_das, replace=False)
        else:
            top_guys = np.sort(self.means[self.notJ])[::-1][:self.n_das].tolist()
            Q_t =  np.random.choice(np.array(self.notJ), self.n_das, replace=False)
            self.N_t = list(set(self.N_t).union(Q_t))
            self.N_t = [arm for arm in self.N_t if arm not in self.J]
            print("self.N_t",self.N_t)
            self.best_arm = self.compute_bt()
            indices_bt = [self.B_ij(a, self.best_arm) for a in self.N_t]
            top_n_das = np.sort(self.means[self.N_t])[::-1][:self.n_das] 
            print("top_n_das",top_n_das)
            self.N_t = [sample(np.argwhere(self.means == max_means).flatten().tolist(), 1)[0] for max_means in top_n_das] 
            print("self.N_t",self.N_t)
        nt = self.compute_nt()
        self.worst_arm = nt
        self.best_arm = self.compute_bt()
        indices_bt = [self.B_ij(a, self.best_arm) for a in self.N_t]
        self.challenger = self.N_t[self.randf(indices_bt, np.max)]
        self.c_t = self.compute_ct()
        print("J_t means", self.means[self.J])
        print("self.B_ij(self.challenger, self.best_arm)",self.B_ij(self.best_arm,self.worst_arm),"bt,st"+str(self.B_ij(self.best_arm,self.challenger)),"st,bt"+str(self.B_ij(self.challenger, self.best_arm)),"nt,challenger"+str(self.B_ij(self.worst_arm,self.challenger)),"challenger,nt"+str(self.B_ij(self.challenger,self.worst_arm)))
        return self.sampling_rule()


    def reward(self, score):
        return np.random.normal(score, self.sigma)


    def get_reward(self, arm):
        ## File IO error if several threads use the same file
        if (False):
            with open(self.scores_filepath, "r") as readFile:
                reader = csv.reader(readFile, delimiter=',')
                for row, record in enumerate(reader):
                    if (row == arm):
                        break
            try:
                record
            except:
                print((self.scores_filepath))
                print((os.path.exists(self.scores_filepath)))
                print("Resource is already used by another thread")
                raise ValueError
        else:
            record = [self.oracle[arm]]
        return self.reward(float(record[0]))


    def play(self, arm, train_data):
        assert arm in self.arms
        #print(self.subsets,self.subsets[14].shape,arm)
        avg_reward, rewards  = LLM_error_indicator(self.subsets[arm],self.val_data) 
        #convert reward to relavance score w/ bert score or something else
        self.rewards.append(avg_reward)
        self.pulled_arms.append(arm)
        #self.data_dict[arm] = create_train_data(self.subsets[arm], rewards,train_data, self.train_emb, self.val_emb)
        self.D.extend(create_train_data(self.subsets[arm], rewards,train_data, self.train_emb, self.val_emb))
        #add qids to list of qids
        #add features to list of features


    def learn_from_arms(self, candidates, train_data): 
        print("``````````Learn from arms")
        if (len(candidates) > 0):
            if (len(candidates) > 0):
                for arm in candidates:
                    print("Inside for loop",candidates)
                    self.play(arm, train_data) 
                    print("~~~~~~~~~~~~~~~~~~~ SELF.T",self.t)
                    self.t += 1
                    self.na[arm] += 1
                print("candidates",candidates,self.rewards,self.rewards[-1])
                self.update(candidates) 



    def initialization(self, T_init=None):
        print("initialized")
        if (str(T_init) == None):
            T_init = self.T_init
        candidates = [[arm]*T_init for arm in self.arms]
        candidates = [y for x in candidates for y in x]
        assert len(candidates) == T_init*self.K
        self.learn_from_arms(candidates)
        


    def recommend_rule(self):
        idx = [int(id_) for id_ in self.m_maximal(self.means.tolist(), self.m)]
        means = self.means[idx].tolist()
        assert all([m >= self.m_max(self.means.tolist(), self.m) for m in means])
        return idx


    def recommend(self):
        idx = self.recommend_rule()
        idx = list(map(int, idx))
        means = self.means[idx].tolist()
        assert all([i in self.arms for i in idx]) and len(idx) == self.m
        assert len(means) == self.m
        return idx, means


    def update_linear_means(self, candidates):
        self.model.train()
        #self.model.to(device)
        #data = dicts_to_tuples(self.data_dict,candidates)
        train_set = MSLRDataset(500, self.num_features, rows = self.D)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        total_loss = 0
        for features, labels, qids in train_loader:
            features, labels = features.to(device), labels.to(device)
            self.optimizer.zero_grad()
            print("featuresfeatures",features.shape,labels.shape)
            scores = self.model(features)
            loss = pirank_simple_loss(labels, scores)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.model.eval()
        preds = []
        with torch.no_grad():
            for (q_id, features,score) in self.pred_data:
                scores = self.model(features.to(device))
                preds.append(scores.item()) #should this data contain each subset or not?
        preds = [preds[i:i + self.list_size] for i in range(0, len(preds), self.list_size)]
        self.means = np.mean(np.array(preds), axis = 0)
        """
        for i in range(1, (self.K*self.T_init if (self.T_init > 0 and self.t <= self.K*self.T_init) else len(candidates))+1):
            x = self.X[:, self.pulled_arms[-i]]
            self.B_inv = self.iterative_inversion(self.B_inv, x)
            self.b += self.rewards[-i]*np.array(x.tolist())[:,0]
            self.theta = np.dot(self.B_inv, self.b).reshape((1, self.N))
            self.means = np.array(np.dot(self.theta, self.X).tolist()[0])
        """
        #print("alpha training error*****************",sum(self.rewards-self.means))
        print("-----***------------updated theta-------------------theta,means",self.theta,self.means)


    def update(self, candidates):
        print("-----------------Going to update theta-------------------")
        self.update_linear_means(candidates)


    def tau(self):
        st_bt = float(self.B_ij(self.challenger, self.best_arm))
        st_nt = float(self.B_ij(self.challenger, self.worst_arm))
        #st_nt = float(self.means[self.challenger] - self.means[self.worst_arm])
        return st_bt, st_nt


    def stopping_rule(self):
        self.B_st_bt, self.B_st_nt = self.tau()
        print("self.B <= self.epsilon",self.B_st_bt,self.B_st_nt,(self.B_st_bt <= self.epsilon) and (self.B_st_nt <= self.epsilon))
        return ((round(self.B_st_bt,2) <= self.epsilon))

    def update_feature_matrix(self, X_new):
        self.X = X_new

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
def create_train_data(subset,  rewards, train_data, train_emb, val_emb):
    #data = {'score': [], 'q_id':[], 'features': []}
    l = []
    #data = {'q_id':[], 'features': []}
    indexes = subset['index'] #convert subsets to indexes
    scores = rewards
    bools = [False if x not in indexes else True for x in range(len(train_data))]
    #bools = np.array(bools)
    features = train_emb[bools]
    for i, score in enumerate(scores):
        mean = np.mean(np.vstack((features,np.expand_dims(val_emb[i], axis = 0))), axis = 0) #convert to pytorch
        l.append((mean,score,int(i)))
        
    return l
        
def dicts_to_tuples(D, keys, use_longest=False, fillvalue=None):
    """
    Convert a dict of dicts-of-lists into a list of tuples.

    Parameters
    ----------
    D : dict[str, dict[str, list]]
        The outer dict whose values are inner dicts mapping to lists.
    use_longest : bool
        If False, uses built-in zip (stops at shortest list).
        If True, uses zip_longest (fills shorter lists with `fillvalue`).
    fillvalue : any
        Value to use for padding if use_longest=True.

    Returns
    -------
    list[tuple]
        A flat list of tuples, one per “row” of the inner lists.
    """
    result = []
    for (k,d) in D.items():
        if k in keys:
            lists = list(d.values())
            if use_longest:
                rows = zip_longest(*lists, fillvalue=fillvalue)
            else:
                rows = zip(*lists)
            result.extend(tuple(row) for row in rows)
        else:
            continue
    return result    

def generate_train_data(df,train_data,train_emb,val_emb):
    #arr = np.zeros(((len(df)*30),768))
    l = []
        #data = {'q_id':[], 'features': []}
    for i in range(len(df)):
        indexes = ast.literal_eval(df['indexes'][i])
        scores = ast.literal_eval(df['scores'][i])
        bools = [False if x not in indexes else True for x in range(len(train_data))]
        
        #bools = np.array(bools)
        features = train_emb[bools]
        for k, score in enumerate(scores[:35]):
        #for i in range(50,60):
            mean = np.mean(np.vstack((features,np.expand_dims(val_emb[k], axis = 0))), axis = 0)
            #arr[(i*30)+k] = mean
            l.append((mean,score,int(k)))
            
    return l

def find_common_sublists(indexes1, indexes2, train_data, L):
    # Build a mapping from tuple-form of each sublist in L1 to all its indices
    idx_map = defaultdict(list)
    for i, sub in enumerate(indexes1):
        idx_map[tuple(sub)].append(i)

    # Check each sublist in L2 against the map
    common = []
    for j, sub in enumerate(indexes2):
        t = tuple(sub)
        if t in idx_map:
            for i in idx_map[t]:
                common.append((i, j))
    
    print(f"Common subset indices:{common}")
    
    if len(common) > 0:
        for (i,j) in common:
            subset = []
            for name, group in train_data.groupby('cluster'):
                # subset.append(group.sample(k//num_gr))   
                subset.append(group.sample(1))   
            subsets = pd.concat(subset)
            L[j] = subsets
            indexes2[j] = subsets['index'].tolist()
    
    return L, indexes2

def construct_L(L_ind,train_data):
    L = []
    for idxs in L_ind:
        subset = []
        for i in idxs:
            subset.append(train_data.loc[train_data['index'] == i])
        subsets = pd.concat(subset)
        L.append(subsets)
    return L

def static_subset_selection(val_data, train_data, k, test_data,filename, inference = False, inf_model = None):

    with open('data/aquarat_train_emb.pkl', 'rb') as f:
        train_emb = pickle.load(f)
         
    with open('data/aquarat_val_emb.pkl', 'rb') as f1:
        val_emb = pickle.load(f1)
        
    val_data = val_data[:50]
    val_emb = val_emb[:50]
    
    val_emb = torch.Tensor(val_emb)
    train_emb = torch.Tensor(train_emb)

    print("*****Embeddings loaded*****")
    # k-means clustering on train_data with k=5
    kmeans = KMeans(n_clusters=5, random_state=0).fit(train_emb)
    
    # Make clusters of train_data with same cluster label
    train_data['cluster'] = kmeans.labels_

    # # Do stratified sampling from train_data based on the first word of the question
    # train_data['w1'] = train_data['question'].str.split().str[0]

    # Create index column in train_data using arange
    train_data['index'] = np.arange(len(train_data))

    # Create L=_ subsets of size k total, with each group having k/num_gr examples
    num_gr = len(train_data['cluster'].unique())
    
    L = []
    L_indices = []

    # print("*****Initializing L*****")
    # Initialize L, 100 random set of subsets from train_data 
    for idx1 in range(100):
        subset = []
        for name, group in train_data.groupby('cluster'):
            # subset.append(group.sample(k//num_gr))   
            subset.append(group.sample(1))   
        subsets = pd.concat(subset)
        L.append(subsets)
        L_indices.append(subsets['index'].tolist())
        
    subsets = pd.DataFrame({"L_ind":L_indices})
    subsets.to_csv("output/aquarat_subsets_mistral3.csv")
    
    #df = pd.read_csv("output/aquarat_subsets.csv", converters={'L_ind': literal_eval}).to_dict(orient = "list")
    #L_indices = df['L_ind']
    L = construct_L(L_indices,train_data)

    #Change for specific llm model
    df = pd.read_csv("data/AQUA_RAT/subset_scores_aqua_mistral3.csv",sep = '\t', converters={'indexes': literal_eval}).to_dict(orient = "list")
    indices1 = df['indexes']
    
    
    L, L_indices = find_common_sublists(indices1, L_indices, train_data, L)
    E_val = cosine_similarity(train_emb, val_emb)


    # ########### Making feature matrix of dim = (num_of_arms, num_of_ices).T
    len_L = len(L)
    L_val = np.zeros((len_L,len(L_indices[0])))
    for u in range(len_L):
        avg = []
        for j in L_indices[u]:
            avg.append(np.mean(E_val[j]))
        avg = np.array(avg).reshape(1,-1)
        L_val[u] = avg

    N = len(L_indices[0])# val_data.shape[0]
    K = len(L)
    X = np.matrix(L_val.T) #np.matrix(np.eye(N,K))#np.random.normal(0, 0.05, size=(N, K))
    #print("X",X,X.shape) 
    
    
    list_size = 500
    batch_size = 16
    num_features = 768
    
    #DELETE COMMENT FOR PRE-TRAINING
    #Change for specific llm model
    train_file = 'data/AQUA_RAT/train_aqua_mistral3.txt'# Training dataset
    val_file = 'data/AQUA_RAT/vali_aqua_mistral3.txt'

    
    #Load the full training dataset
    train_set = MSLRDataset( list_size, num_features,file_path=train_file)

    val_set = MSLRDataset(list_size, num_features,file_path=val_file)

    #Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model, infos, optimizer = train_ltr_model(train_loader, val_loader, num_features, num_epochs=100, learning_rate=0.001,filename = filename)
    
    #COMMENT UNTIL AND INCLUDING LINE 959(loss = checkpoint['loss']) FOR PRE-TRAINING
    # model = RankNet(num_features).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # checkpoint = torch.load("output/case_rank_aquarat2_model.pth", weights_only=True) 
    # model.load_state_dict(checkpoint)
    
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    

    
    
    top_m_instance = top_m_arm(args={},X=X,
                               m=10,
                               model = model,
                               optimizer=optimizer,
                               val_data=val_data[:20],
                               subsets=L)
    top_m_instance.clear()
    top_m_instance.val_emb = val_emb
    top_m_instance.train_emb = train_emb
    top_m_instance.L_ind = L_indices
    
    df = pd.read_csv("data/AQUA_RAT/subset_scores_aqua_mistral3.csv", sep = '\t')    #Change for specific llm model
    top_m_instance.D = generate_train_data(df,train_data,train_emb,val_emb)
    
    l = []
    for i in range(val_emb[:20].shape[0]):
        l.extend(generate_test_input(i,L_indices,train_emb,val_emb))
    top_m_instance.pred_data = l #Do I need to pad test input data as well since list size is fewer?
    #top_m_instance.initialization(T_init=top_m_instance.T_init)

    if inference == True:
        model = RankNet(768).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
        checkpoint = torch.load(inf_model, weights_only=True) #change wıth a model whose name end with "model"
        model.load_state_dict(checkpoint['model_state_dict'])
        top_m_instance.model = model
        return top_m_instance

    i=0
    done=False
    round_mse = []
    acc_dict={}
    true_means = []
    arm_list = []
    top_m_instance.patience = 0
    
    while not done or top_m_instance.patience< 5:
        print(("** Round #"+str(i+1)), end=' ')
        i += 1
        candidates = top_m_instance.sample()
        top_m_instance.learn_from_arms(candidates,train_data)
        done = top_m_instance.stopping_rule()
        '''
        if(i == 1):
            print("Will be calculating accuracy for all arms in J (val_data * J)llm calls")
            #true_means = [LLM_error_indicator(L[arm_i], val_data) for arm_i in top_m_instance.J]
            for arm_i in top_m_instance.J:
                if arm_i not in acc_dict:
                    acc_dict[arm_i] = LLM_error_indicator(L[arm_i], val_data)
                true_means.append(acc_dict[arm_i])
                arm_list.append(arm_i)
                
        else:
            if top_m_instance.c_t not in acc_dict:
                acc_dict[top_m_instance.c_t] = LLM_error_indicator(L[top_m_instance.c_t], val_data)

            # need to get proper worst index of J
            true_means.pop(top_m_instance.J.index(top_m_instance.worst_arm))
            arm_list.pop(top_m_instance.J.index(top_m_instance.worst_arm))

            # it is fine but J_t_means won't be having the last index as ct_mean as we are sorting J after appending ct
            # true_means.append(LLM_error_indicator(L[top_m_instance.c_t], val_data))

            true_means.append(acc_dict[top_m_instance.c_t])
            arm_list.append(top_m_instance.c_t)

        #J_t_means = [top_m_instance.means[arm_1] for arm_1 in top_m_instance.J]
        J_t_means = [top_m_instance.means[arm_1] for arm_1 in arm_list]
        #true_means = [true_means[arm_1] for arm_1 in top_m_instance.J]

        print("\n*************Approximation error of Validation Data on J ************")
        mse = ((np.array(J_t_means) - np.array(true_means))**2)
        round_mse.append(mse)
        print("\n mse between accuracies ", mse) 
    
    result = top_m_instance.recommend() #bu direk kaldirilicak
    print("result*******",result)
    best_arms = result[0]
    best_arms_subsets = [L[k] for k in best_arms]


    #===============================================================================
    # Calculate the pairwise overlap between the subsets in U
    overlaps=[]
    for i in range(len(best_arms_subsets)):
        inner_overlaps=[]
        for j in range(len(best_arms_subsets)):
            if i!=j:
                overlap=0
                for index_j, s_i in best_arms_subsets[i].iterrows():
                    for index_j, s_j in best_arms_subsets[j].iterrows():
                        if s_i["question"].lower() in s_j["question"].lower() or s_j["question"].lower() in s_i["question"].lower():
                            overlap+=1
                inner_overlaps.append(overlap)
        overlaps.append(inner_overlaps)
            
    print("\nOverlaps:",overlaps)
    print("Len of overlaps:",len(overlaps))

    overlap_for_subset = []
    avg_overlap = []
    min_overlap = []
    max_overlap = []
    overlap_for_each_subset = np.average(overlaps, axis=1)
    overlap_avg = np.average(overlap_for_each_subset)
    overlap_min = np.min(overlap_for_each_subset)
    overlap_max = np.max(overlap_for_each_subset)
    overlap_for_subset.append(overlap_for_each_subset.tolist())
    avg_overlap.append(overlap_avg.tolist()) 
    min_overlap.append(overlap_min.tolist())
    max_overlap.append(overlap_max.tolist())
    print("\n********* PAIRWISE OVERLAP *********")
    print("\nOverlap_for_subset:",overlap_for_subset)
    print("\nAVG_overlap:",avg_overlap)
    print("MIN_overlap:",min_overlap)
    print("MAX_overlap:",max_overlap)
    '''
    #===============================================================================
    result = top_m_instance.recommend()
    print("result*******",result)
    best_arms = result[0]
    best_arms_subsets = [L[k] for k in best_arms]
    folder1 = f"./output/loss_folder"
    #np.savez(f'{folder1}', overlap_for_subset = overlap_for_subset , avg_overlap = avg_overlap, min_overlap = min_overlap, max_overlap = max_overlap, round_mse = round_mse)
    
    return top_m_instance, best_arms_subsets

def generate_feature_matrix(train_emb, test_emb_inst, L_indices, L):
    E_val = cosine_similarity(train_emb, torch.unsqueeze(test_emb_inst, dim = 0))


    # ########### Making feature matrix of dim = (num_of_arms, num_of_ices).T
    len_L = len(L)
    L_val = np.zeros((len_L,len(L_indices[0])))
    for u in range(len_L):
        avg = []
        for j in L_indices[u]:
            avg.append(np.mean(E_val[j])) #you do not need to take average
        #avg = avg/len(L_indices[u])
        avg = np.array(avg).reshape(1,-1)
        L_val[u] = avg
        
    N = len(L_indices[0])# val_data.shape[0]
    K = len(L)
    X = np.matrix(L_val.T) #np.matrix(np.eye(N,K))#np.random.normal(0, 0.05, size=(N, K))
    return X, K, N

def get_best_subset(subsets, scores,k=10):
    ranked_docs = sorted(zip(subsets, scores), key=lambda x: x[1])
    best_subsets = [x[0] for x in ranked_docs[:k]]
    return best_subsets   

def get_predictions(model,test_input):
    model.eval()
    scores = []
    with torch.no_grad():
        for (q_id, features,score) in test_input:
            features = features.to(device)
            scores.append(model(features))
            
    return scores
            
            
    
            
    

def get_open_source_completions(test_data, data, file_name):

    # stop_signal = "\n\n"
    matches = 0
    mismatches = 0

    question_df = {"question":[],"answers":[]}
    
    train_data, val_data = train_test_split(data, test_size=0.3, random_state=42)


    with open('data/AQUA_RAT/aquarat_test_emb.pkl', 'rb') as f2:
        test_emb = pickle.load(f2)
        
    test_emb = torch.Tensor(test_emb)
    
    
    print("*************Static subset selection*************")    
    #INPUT FALSE FOR INFERENCE PARAMETER TO RUN BANDIT ALGORITHM
    trained_model = "models/Trained_models/case_rank_aquarat_mistral3_model.pth"
    top_m_instance, subsets = static_subset_selection(val_data, train_data, 5, test_data,file_name, inference=False, inf_model=trained_model)
    model = top_m_instance.model
    merged_exemplars = pd.concat(subsets)
    merged_exemplars.to_csv("GRASS_aqua_U_t.csv")
    torch.save(model.state_dict(),f"output/{file_name}_model.pth" )

    print("*************Finished static subset selection*************")

    index=0
    acc_records = []
    merged_exemplars = {}
    exnum = 1
    for index,row in tqdm(test_data.iterrows(),total = test_data.shape[0],desc="Generating"):
        print(f"*************TEST INSTANCE NO:{exnum}*************")
        test_input = generate_test_input(exnum-1,top_m_instance.L_ind,top_m_instance.train_emb,test_emb)
            
        scores = get_predictions(model,test_input)
        
        exemplars = get_best_subset(top_m_instance.subsets, scores)
        #final_exemp = [exemplar.to_json() for exemplar in exemplars[0]]
        merged_exemplars[exnum] = exemplars[0].to_json()

        print("\n\n\n********************Take the exemplar with minimum validation loss and use it as the exemplar")


        #avg_err = LLM_avg_error(exemplars, val_data)
        #print("\n\nAvg Error:",avg_err)
        #ind = np.argmin(avg_err)
        #print("\n\nMin index:",ind)
        #exemplars = exemplars[ind]
        prompt = prompt_for_manual_prediction(row, exemplars[0])
        tmp = llm_output(prompt)
        #print(len(tmp_list))
        #n = self_con(tmp_list)
        answer = extract_option(tmp)

        gt = row["correct"]
        # print("GT: ", gt)
        if(answer==gt):
          matches+=1
        else:
          mismatches+=1

        print("\nAnswer:", answer)
        print("Ground Truth:", gt)

        question_df['question'].append(row["question"])
        question_df["answers"].append(answer)
        
        print("Accuracy:", matches/exnum)
        print("EM:", matches/(matches+mismatches))
        exnum += 1
        try:
            with open(f'output/{file_name}_examplars.json', "w") as file:
                json.dump(merged_exemplars, file, indent=4)
            print("Data successfully saved to merged_examplars.json")
        except IOError as e:
            print(f"Error saving data: {e}")
            
        final_questions = pd.DataFrame(question_df)
        final_questions.to_csv(f"output/{file_name}.csv",sep="\t",index=False)
    return final_questions

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_few_shot_prediction():

    ### Load data
    # with open("../data/GSM8K/gsm8k_train.jsonl", 'r') as f:
    #     json_list = list(f)
    # train_set = [json.loads(x) for x in json_list]

    train_set = pd.read_csv("data/AQUA_RAT/aquarat_train.csv")

    test_set = pd.read_csv("data/AQUA_RAT/aquarat_dev.csv")
    
    final_df = get_open_source_completions(test_set, train_set, "GRASS_aquarat")
    print(final_df)


if __name__=='__main__':
    test_few_shot_prediction()
