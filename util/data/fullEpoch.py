import json
import numpy as np

def cartesian(data1, data2):
    result = [[i, j] for i in data1 for j in data2]
    result = np.array(result)
    return result

def fullEpoch(data):
    result = cartesian(data, data)
    np.random.shuffle(result)
    result = result.tolist()
    res1 = [i[0] for i in result]
    res2 = [i[1] for i in result]
    return res1, res2

if __name__ == '__main__':
    pass

