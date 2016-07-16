def main():
    sum_for = 0
    sum_while = 0
    for x in range(1,101):
        if x % 2 == 0:
            sum_for += x
    x = 1;
    while x <= 100:
        if x % 2 == 0:
            sum_while += x
        x += 1
    print('the for loop\'s sum is ' + str(sum_for))
    print('the while loop\'s sum is ' + str(sum_while))

if __name__ == '__main__':
    main()