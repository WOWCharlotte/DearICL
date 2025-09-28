import pandas as pd
import re

def extract_option(output: str) -> str:
    matches = re.findall(r'([A-E]\) |[A-E])', output) #'([A-E]\))'
    return matches[-1].strip(')')[0] if matches else None


def evaluate_aqua(gt, predictions):
    matches = 0
    mismatches = 0
    for index, prediction in enumerate(predictions):
        pred = extract_option(prediction)
        if pred == gt[index]:
            matches+=1
        else:
            mismatches+=1
    return matches/(matches+mismatches)







if __name__ == "__main__":
    predictions = pd.read_csv("../../output/GRASS_aquarat_lalam_3_2_test.csv", sep="\t")
    ground_truth = pd.read_csv("../../data/AQUA_RAT/aquarat_dev.csv")

    gt = list(ground_truth["correct"].values)
    predictions = list(predictions["answers"].values)
    print(evaluate_aqua(gt, predictions))