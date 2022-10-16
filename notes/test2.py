def func():
    n=0
    while True:
        res = yield n
        if res:
            n= res
        n+=1
g=func()
print(g.send(None))
print(g.send(3))
print(next(g))