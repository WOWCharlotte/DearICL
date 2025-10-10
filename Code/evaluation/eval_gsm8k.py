import pandas as pd
import json
def clean_ans(s):
    ans_s = s.split("#### ")[1]
    ans_s = ans_s.replace(",","")
    return ans_s
if __name__ =="__main__":
    data = pd.read_csv("../../output/case_rank_gsm8k_final_v3.csv", sep="\t")

    with open("../../data/GSM8K/gsm8k_test.jsonl", 'r') as f:
        json_list = list(f)
    test_set = [json.loads(x) for x in json_list]
    matches = 0
    mismatches = 0
    for index, sample in enumerate(test_set):
        print(index)
        answer = data.iloc[index,:]["answers"]
        gt = int(clean_ans(sample["answer"]))
        if answer==answer and int(answer)==gt:
            matches+=1
        else:
            mismatches+=1
    print("EM",matches/(matches+mismatches))

