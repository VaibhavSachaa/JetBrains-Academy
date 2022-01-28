# write your code here
# s = input('Enter cells: ')
val = [' '] * 9

player_1 = 'X'
player_2 = 'O'


def grid(value):
    mat = []
    print('---------')
    for i in range(0, 3):
        mat.append([])
        for j in range(0, 3):
            mat[i].append(value[3 * i + j])
            if j == 0:
                print('|', mat[i][j], end=' ')
            elif j == 2:
                print(mat[i][j], '|')
            else:
                print(mat[i][j], end=" ")
    print('---------')


grid(val)

flag = True
turns_left = 9
while turns_left >= 1:
    for i in range(2):
        while flag:
            row, col = input("Enter the coordinates: ").split(" ")
            if not (row.isdigit() and col.isdigit()):
                print("You should enter numbers!")
                continue

            row = int(row)
            col = int(col)

            if not (3 >= row >= 1 and 3 >= col >= 1):
                print("Coordinates should be from 1 to 3!")
                continue

            if val[3 * (row - 1) + (col - 1)] != ' ':
                print("This cell is occupied! Choose another one!")
                continue
            else:
                if i == 0:
                    val[3 * (row - 1) + (col - 1)] = player_1
                else:
                    val[3 * (row - 1) + (col - 1)] = player_2
            turns_left = turns_left - 1
            # print(turns_left)
            # print(val)
            flag = False
            grid(val)

        if turns_left <= 4:
            s = ''.join(val)
            # print(s)
            result = ''
            if ('XXX' in s[::3] or 'XXX' in s[1::3] or 'XXX' in s[2::3]) and (
                    'OOO' in s[::3] or 'OOO' in s[1::3] or 'OOO' in s[2::3]):
                result = 'Impossible'
                print('Impossible')
            elif ('XXX' in s and 'OOO' in s) or (abs(s.count('X') - s.count('O')) >= 2):
                result = 'Impossible'
                print('Impossible')
            elif 'XXX' in s[::3] or 'XXX' in s[1::3] or 'XXX' in s[2::3]:
                result = 'X wins'
                print('X wins')
            elif 'OOO' in s[::3] or 'OOO' in s[1::3] or 'OOO' in s[2::3]:
                result = 'O wins'
                print('O wins')
            elif 'XXX' in s[:3] or 'XXX' in s[3:6] or 'XXX' in s[6:]:
                result = 'X wins'
                print('X wins')
            elif 'OOO' in s[:3] or 'OOO' in s[3:6] or 'OOO' in s[6:]:
                result = 'O wins'
                print('O wins')
            elif 'XXX' in s[::4] or 'XXX' in s[2:7:2]:
                result = 'X wins'
                print('X wins')
            elif 'OOO' in s[::4] or 'OOO' in s[2:7:2]:
                result = 'O wins'
                print('O wins')
            elif ' ' not in s:
                result = 'Draw'
                print('Draw')
            # print(s)
            if result != "":
                exit()
        if turns_left == 0:
            flag = False
        else:
            flag = True
