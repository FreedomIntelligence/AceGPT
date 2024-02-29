import json
import os

def cal_acva_f1(result_path):
    subtasks = ['Arabic Funeral', 'Sudan', 'Arabic Physics and Chemistry', 'Algeria', 'InfluenceFromAncientEgypt', 'Arabic Ceremony', 'Arabic Astronomy', 'Arabic Calligraphy', 'daily life', 'Saudi Arabia', 'Arabic Language Origin', 'Arabic Ornament', 'Islamic law system', 'Kuwait', 'InfluenceFromChina', 'Arabic Literature', 'computer and phone', 'Tunisia', 'Arabic Geography', 'Arabic Music', 'Arabic Medicine', 'Arabic Philosophy', 'Yemen', 'Jordan', 'Mesopotamia civilization', 'Islam Education', 'Arabic Wedding', 'InfluenceFromRome', 'Egypt modern', 'Ancient Egypt', 'Comoros', 'InfluenceFromGreece', 'Qatar', 'InfluenceFromPersia', 'Lebanon', 'Arabic Math', 'Arabic History', 'InfluenceFromIslam', 'Libya', 'Syria', 'Oman', 'Arabic Culture', 'Arabic Art', 'United Arab Emirates', 'Islam branches and schools', 'InfluenceFromByzantium', 'Arab Empire', 'Arabic Food', 'Mauritania', 'entertainment', 'communication', 'Palestine', 'Bahrain', 'Somalia', 'Iraq', 'Arabic Clothing', 'Arabic Architecture', 'Morocco']
    n_all = 0
    n_yes = 0
    n_pred_yes = 0
    n_correct_yes = 0
    n_no = 0
    n_pred_no = 0
    n_correct_no = 0
    for subtask in subtasks:
        data = []
        file_path = os.path.join(result_path, f'{subtask}.jsonl')
        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file, 1):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"Error decoding JSON on line {line_number} of {file_path}")

        n_all += len(data)
        n_yes += sum(x['answer'] in ( "نعم") for x in data)
        n_pred_yes += sum(x['response_answer'] in ( "نعم") for x in data)
        n_correct_yes += sum(x['response_answer'] in ( "نعم") and x['answer'] in ("نعم") for x in data)
        n_no += sum(x['answer'] in ( "لا") for x in data)
        n_pred_no += sum(x['response_answer'] in ( "لا") for x in data)
        n_correct_no += sum(x['response_answer'] in ("لا") and x['answer'] in ( "لا") for x in data)
    P_no = n_correct_no / n_pred_no
    R_no = n_correct_no / n_no
    F1_no = 2 * P_no * R_no / (P_no + R_no)
    P_yes = n_correct_yes / n_pred_yes
    R_yes = n_correct_yes / n_yes
    F1_yes = 2 * P_yes * R_yes / (P_yes + R_yes)
    avg_P = (P_no + P_yes) / 2
    avg_R = (R_no + R_yes) / 2
    avg_F1 = (F1_no + F1_yes) / 2
    print(f'avg_F1 = {round(avg_F1, 4)}')
    return avg_F1

if __name__=="__main__":
    result_path =  ''
    avg_F1 = cal_acva_f1(result_path)