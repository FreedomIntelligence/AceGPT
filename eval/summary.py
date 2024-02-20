import json
import numpy as np
import collections

"""MMLU score"""
def get_average_score_for_categories_new(data):
    results = collections.defaultDict(int)
    
    for category, subcategories in data.items():
        for subcategory, values in subcategories.items():
          results[category] += values["average"]["Accuracy"]
    results[category] /= len(subcategories)        
    score = np.mean([average for category, average in results.items()])
    for category, average in results.items():
      print(f"{category}: {average:.2%}")
    print(f'mean of all category: {score:.2%}')
    print([float(f'{score*100:.2f}')]+[float(f'{ave*100 :.2f}') for ave in results.values()])
    return results,score
