def splitByUpper(name: str):
    tokens = []
    i = 0
    for j, c in enumerate(name):
        if c.isupper():
            tokens.append(name[i:j])
            i = j
    tokens.append(name[i:])
    return (" ".join(tokens)).title()


if __name__ == '__main__':
    print(splitByUpper("hello"))
    print(splitByUpper("randomAgent"))
    print(splitByUpper("thisIsSomeClass!"))
