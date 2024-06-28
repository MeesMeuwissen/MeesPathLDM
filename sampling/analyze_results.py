import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import sys

def calculate_kappa(ground_truth, answers):
    correct_real = sum([a == b and a == "Real" for a, b in zip(ground_truth, answers)])
    correct_synthetic = sum([a == b and a == "Synthetic" for a, b in zip(ground_truth, answers)])

    incorrect_real = sum([a != b and a == "Real" for a, b in zip(ground_truth, answers)])
    incorrect_synthetic = sum([a != b and a == "Synthetic" for a, b in zip(ground_truth, answers)])

    total = correct_real + incorrect_real + correct_synthetic + incorrect_synthetic

    # Calculate Cohen's Kappa
    p_o = (correct_real + correct_synthetic) / total

    p_real = (correct_real + incorrect_real) / total * (correct_real + incorrect_synthetic) / total
    p_synthetic = (incorrect_synthetic + correct_synthetic) / total * (incorrect_real + correct_synthetic) / total
    p_e = p_real + p_synthetic

    kappa = (p_o - p_e) / (1 - p_e)

    print("Kappa = " + str(kappa))
    print("sklearn kappa:", cohen_kappa_score(ground_truth, answers))
    return kappa, p_o, p_e

def get_kappas(df, ground_truth, experiences):
    kappas = []
    corrects = []
    for index, participant in df.iterrows():
        answers = list(participant[:40])
        kappa = cohen_kappa_score(ground_truth, answers)
        corrects.append(sum([1 if x==y else 0 for x,y in zip(ground_truth, answers)]))


        kappas.append(kappa)
        # print(f"Kappa: {kappa:.2f}")

    #kappas = [float(format(x, '.2f')) for x in kappas]
    print("Kappas:", kappas)
    print(sum(kappas) + 0.04 -0.05 )
    print("Experiences: ", experiences)
    print(f"Mean: {np.mean(kappas)}. Max: {np.max(kappas)}. Min: {np.min(kappas)}")

    for ind, exp in enumerate(experiences):
        print(f"Experience: {exp}. Kappa: {kappas[ind]}. # Correct: {corrects[ind]}")

    print(sum(corrects) / len(corrects))



def get_error_per_question(df, ground_truth):
    synth_results = {}
    real_results= {}
    for question, answers in df.iterrows():
        if ground_truth[question - 1] == 'Real':
            # GT is Real
            real_results.update({question:sum([answer == 'Real' for answer in answers])})
        if ground_truth[question - 1] == 'Synthetic':
            # GT is synthetic
            synth_results.update({question:sum([answer == 'Synthetic' for answer in answers])})
    print(synth_results)

    print(sum(synth_results.values()))
    print(sum(real_results.values()))
    print((sum(synth_results.values()) + sum(real_results.values()))/40)
    # for key in synth_results.keys():
    #     print(f"Question {key} had {synth_results[key]}/{df.shape[1]} correct answers.")
    #
    # for key in real_results.keys():
    #     print(f"Question {key} had {real_results[key]}/{df.shape[1]} correct answers.")



    best_fake = min(synth_results, key=synth_results.get)
    worst_fake = max(synth_results, key=synth_results.get)

    worst_real = min(real_results, key=real_results.get)
    best_real = max(real_results, key=real_results.get)

    print(f"The Synthetic image with the lowest correct answers is question {best_fake} ({synth_results[best_fake]})")
    print(f"The Synthetic image with the highest correct answers is question {worst_fake} ({synth_results[worst_fake]})")

    print(f"The Real image with the lowest correct answers is question {worst_real} ({real_results[worst_real]})")
    print(f"The Real image with the highest correct answers is question {best_real} ({real_results[best_real]})")


def dominant_value(row):
    # Calculates what value occurs more often in a given row, Real or Synthetic
    real_count = (row == 'Real').sum()
    synthetic_count = (row == 'Synthetic').sum()

    if real_count > synthetic_count:
        return 'Real'
    elif synthetic_count > real_count:
        return 'Synthetic'
    else:
        return 'Even Split'

def majority_vote_stuff(df, ground_truth):
    majority_vote = df.apply(dominant_value, axis=1)

    correct = [a==b for a, b in zip(majority_vote, ground_truth)]
    even_split = sum([a == "Even Split" for a in majority_vote])

    print(f"Majority vote split in {even_split} cases.")
    print(f"Majority vote correct in {sum(correct)}/{len(correct)} cases.")

def main():
    datapath = "/Users/Mees_1/MasterThesis/Aiosyn/data/survey_imgs/cleanedResults.csv"
    df = pd.read_csv(datapath)
    # Answers are loaded per question. That means, in this df, all answers to question 1 are in a single row.
    # Transposing it means all answers by a single person are in a row, which makes it easier to analyse it.

    ground_truth = list(df.iloc[:, 0][1:41])

    experiences = list(map(int, list(df.iloc[0,2:]))) # List of all experience levels of the participants.
    df = df.iloc[1:, 2:] # Remove first row, first two cols
    df = df.transpose() # Row = one respondent now

    get_kappas(df, ground_truth, experiences)

    # Calculate error rates per synthetic image:
    df = df.transpose() # Row = one question
    #get_error_per_question(df[:40], ground_truth) # Only 40 questions

    #majority_vote_stuff(df[:40], ground_truth)

if __name__ == '__main__':
    main()