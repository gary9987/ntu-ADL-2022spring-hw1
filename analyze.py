import json


if __name__ == '__main__':
    '''
    f = open('./data/intent/train.json')
    data = json.load(f)
    max_len = 0
    for i in data:
        max_len = max(max_len, len(i['text'].split()))
    print(max_len)

    f = open('./data/intent/eval.json')
    data = json.load(f)
    max_len = 0
    for i in data:
        max_len = max(max_len, len(i['text'].split()))
    print(max_len)

    f = open('./data/intent/test.json')
    data = json.load(f)
    max_len = 0
    for i in data:
        max_len = max(max_len, len(i['text'].split()))
    print(max_len)
    '''
    dic = {}
    f = open('./data/intent/train.json')
    data = json.load(f)
    max_len = 0
    for i in data:
        try:
            dic[i['intent']] += 1
        except:
            dic[i['intent']] = 1
    print(dic)
