import os

# print(os.environ)
# print(os.environ.keys())

print(os.environ.get("PWD"))
print(os.environ.get("USER"))
print(os.environ.get("HOME"))

print(os.getcwd())
print(os.listdir())

if not os.path.exists('test'):
    os.mkdir('test')

os.rename("./test", "test2")

"""
   / : 최상위 경로
  ./ : 현재 경로
 ../ : 상위 경로
"""


