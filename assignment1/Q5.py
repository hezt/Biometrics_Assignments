import random


def InsertSort(seq):
    if len(seq) == 1:
        return seq
    for i in range(1, len(seq)):
        temp = seq[i]
        j = i - 1
        while j >= 0 and temp < seq[j]:
            seq[j + 1] = seq[j]
            j -= 1
        seq[j + 1] = temp
    return seq


def main():    
    testseq = []
    for i in range(20):
        testseq.append(random.randint(1, 200)) 
    print (testseq)
    print (InsertSort(testseq))

if __name__ == '__main__':
    main()
