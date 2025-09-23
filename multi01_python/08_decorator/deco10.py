class Greeting:

    def __init__(self, name):
        self.name = name


    def __call__(self, func):

        def wrapper(*args, **kwargs):
            print(f"Hello {self.name}")
            func(*args, **kwargs)

        return wrapper

@Greeting("pyhton")
def myfunc():
    print("Hello world")

if __name__ == "__main__":
    myfunc()
    # Greeting("python")(myfunc)()
