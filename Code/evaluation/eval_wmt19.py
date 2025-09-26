
import bert_score 
import pandas as pd

scorer = bert_score.BERTScorer(model_type='bert-base-chinese')


def bert_score(ground_truth, predictions):
    # stop_signal = "\n\n"
    error=[]
    exnum = 0
    f=0
    # increment error if model predicted answer is not equal to the ground truth answer for the test question by passing the exemplars
    for index, prediction in enumerate(predictions):       
        
        # acc_records = []
        exnum+=1
        print("prediction",prediction,ground_truth[index])
        if prediction!=prediction:
            continue
        P, R, F1 = scorer.score([prediction], [ground_truth[index]])    
        f+= F1.item()  
        print("EM",f/exnum)                
        


    return f/exnum

if __name__=="__main__":
    test_set = pd.read_csv("../../data/WTM19/wtm19_test.csv")
    predictions = pd.read_csv("../../output/WMT19_results/GRASS_wtm19_inference_test_best.csv",sep="\t")
    ground_truth = list(test_set["zh"].values)
    predictions = list(predictions["zh"].values)
    print("predictions",len(predictions),len(ground_truth))
   
    print(bert_score(ground_truth,predictions))