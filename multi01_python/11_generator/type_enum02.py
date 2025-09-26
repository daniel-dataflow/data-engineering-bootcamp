from enum import Enum


color = Enum("color", ["RED", "BLUE", "GREEN"])

print(color)
print(color.RED)
print(color.RED.name)
print(color.RED.value)
