def subject_generator():
    yield "python"
    yield "numpy"
    yield "pandas"


if __name__ == "__main__":
    subjects = subject_generator()
    print(subjects)

    for subject in subjects: # for문 __next__() 를 내제되어 있다.
        print(subject)

    # print(subjects.__next__())
    # print(subjects.__next__())
    # print(subjects.__next__())
    # print(subjects.__next__())