def main():
    sin = input('Enter a string: ')
    print('You entered: ' + sin)
    l = []
    for char in sin:
        if char == ' ':
            l.append('-')
        else:
            l.append(char)
    sout = ''.join(l)
    print('Result: ' + sout)

if __name__ == '__main__':
    main()
