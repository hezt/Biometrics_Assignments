def main():
    s = input('Enter a string: ')
    print('You entered: ' + s)
    print('Length: ' + str(len(s)))
    print('Upper case: ' + s.upper())
    print('Lower case: ' + s.lower())
    print('Reverse: ' + s[::-1])

if __name__ == '__main__':
    main()
