'''
*
**
***
****
*****
'''
def star01():
    for i in range(6):
        print('*'*i)




'''
*****
****
***
**
*
'''
def star02():
    for i in range(5,0,-1):
        print('*'*i)



'''
    *
   **
  ***
 ****
*****
'''
def star03():
    for i in range(6):
        a = '*'*i
        c = f'{a:>5}'
        print(c)




'''
*****
 ****
  ***
   **
    *
'''
def star04():
    for i in range(5, 0, -1):
        a = '*' * i
        c = f'{a:>5}'
        print(c)




'''
    *
   ***
  *****
 *******
*********
'''
def star05():
    for i in range(10):
        a = '*' * i
        c = f'{a:^10}'
        if i%2 != 0:
            print(c)



if __name__ == '__main__':
    star01()
    star02()
    star03()
    star04()
    star05()
