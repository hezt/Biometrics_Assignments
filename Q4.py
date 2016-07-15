def main():
    L = []
    print(L)
    L.append(5)
    print(L)
    L.append(10)
    print(L)
    L.append(3)
    print(L)
    L.insert(0, 9)
    print(L)
    L_square = L*2
    print(L_square)
    for item in L_square[::]:
        if(item == 10):
            L_square.remove(item)
    print(L_square)
    L_r = L_square[::-1]
    print(L_r)

if __name__ == '__main__':
    main()
