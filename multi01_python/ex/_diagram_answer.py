"""
#####
 ###
  #
*****
*****
*****
  @
 @@@
@@@@@

"""
def diagram1():
    for i in range(3):
        print(" " * i + "#" * (5 - 2 * i))

def diagram2():
    for i in range(3):
        print("*****")

def diagram3():
    for i in range(3):
        print(" " * (2 - i) + "@" * (2 * i + 1))

if __name__ == '__main__':
    diagram1()
    diagram2()
    diagram3()





