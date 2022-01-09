import hashlib
from decimal import *
import random
from functools import reduce
from typing import *
import time
import datetime


class Review:
    """"""
    """
    Pycharm 
        read documentation: click on function, View -> Quick Documentation  (Ctrl+Q)
    
    virtualenv venv   # create env folder for project dependency
        # or python -m venv venv   # create virtual env after python 3.5
        # add environment path to the path variable
    which pip   # shows which environment is installing
    python env/bin/pip install urllib   # force declare which pip
    
    --Naming convention
    variable name, module(file) name: letter number _, can't start with number, case sensitive. module name can use '-' 
    better use student_name than studentName (camel case)
    function name: camel case, first character lowercase       def getName():
    class name, project name: camel case, first character capitalized   class SchoolStudent:
    student_age, age = 3, 3   can declare multiple variable at same time
    a, b = b, a    swap a and b
    id(object)  # check rendered memory location 
    
    --Data type
    immutable data types: int, float, decimal, bool (True False), string, tuple, and range.
        immutable data type when value change, will create a new item in memory if that value doesn't already in memory,
        and assign new location to variable  
        so a=1;   b=a;   a=2;  print(b) # 1   
    mutable data types: list, dictionary, set and user-defined classes.
        mutable data type when value change, will not affect the location it stored in
        li = [1,2];  li2=li;  li[0]=0;  li.append(3);  print(li2) # [0,2,3]
    data type can be changed once the value has been changed
    pi = Decimal('1.1')  # used to avoid inaccuracy   float might get 1.1000000000000001
    type(pi)  # <class 'Decimal'>  return data type of pi
    
    
    --Print and input
    print(object(s), sep=separator, end=end, file=file, flush=flush)
    print(a, b)  # default space separate variables, default new line use sep=',' to add separator  end='' no new line
    print(a,b,sep=',',end='')
    input(prompt string) allows user input. program pause until input received. always return string object
    
    
    --String
    quotes inside string need alternative double and single quotes or use \", \'  word = "He's cool"
    use r"C:\path"   so it won't transform items like \t
    use '''    ''' to keep format (spacing)
    casting to string use str()   ex: str(age)
    String Formatting  print("%s is %d years old." % (name, age))  %%  %s  %d  %f  %c  %E
        float can use %s to convert to string   %.2f  keep 2 decimal
        print("{} is {} years old.".format(name, age))
        print("{0} is {1} years old. I am {1} years old".format(name, age))
        print("{name} is {age} years old.".format(name='Harry', age=10))
        name='Harry' 
        print(f"{name} is nice")    # name variable is defined ahead
        values = {'city': 'San Francisco', 'state': 'California'}
        s = "I live in %(city)s, %(state)s" % values    # I live in San Francisco, California

    variable with same string value share the same memory location (reference), but change one won't affect other
    slicing: s[start:end:step]  [start,end) :step is optional.   start, end can be negative, last char has index -1. 
        start, end can be empty, default to 0 and last index
    functions:
    len   # len(var)  string length
    find    index    rfind   rindex     # var.find('_') return index of first _, return -1 if not found
                                        # var.find('_',0,5) return index of first _ between 1st and 5th char
                                        # var.index('_')  same as find, but will raise exception if not found
                                        # var.index('_', beg=0, end=len(string))
    startswith    endswith   isalpha  isdigit  isalnum   isspace   # var.startswith('www.') return bool  
                                        # var.isalnum()  check for whether is alphabets or num
    count                               # var.count('_')  count _ occurrence times
    replace                             # var.replace('old', 'new', 2)  replace substring old to new for max 2 times, 
                                        # 2 is optional, default is change all occurrence
    split  rsplit  splitlines  partition  rpartition   # var.split(' ', 1)  return list of substring split by space and 
                                        # max split one time, 1 is optional, default to all occurrence
                                        # var.splitlines()  split every line of string
                                        # var.partition(' ') return a tuple with left substring, first space, and 
                                        # right substring
    capitalize title  upper lower       # var.title()  return every word first character capitalized  
                                        # var.capitalize()    return first character capitalized  
    ljust  rjust  center  lstrip  rstrip  strip  replace   # var.lstrip()  remove left spaces before 1st non-space char
                                        # var.center(30)   return 30 length string with original at center, rest spaces
                                        # var.rjust(30)  return 30 length string with original at right, rest spaces                                               
    join                                # ' '.join(iterable)   return string of contacting elements with space between
    
    a ='i'    print(a =='i')  # True
    print(a +'j')   # ij , won't change a value, since python string is immutable
    a += 'abc'  # naive string concatenate O(n^2), since create 3, 6, 9, ...  length string
        li =[]   for i in range(count): li.append('abc')   res = ''.join(li)
        or ''.join(['abc' for i in range(count)])     # O(n)
        
    chr(65)    # return string of unicode       ord('A')   # return unicode of character
    ord('c') - ord('a')   # 2
         
    --Operators
    arithmetic operator: +  -  *  /  **  //  %  
    assignment Operators: =  +=  -=  *=  /=  %=  //=  **=  &=  |=  ^=  >>=  <<=
    comparison operator: >  <  >=  <=  ==  !=  
    identity operator: is  is not
    membership operator: in  not in
    logical operator: and  or  not     print(1 and 3)  # 3
    bitwise operator:  &   |   ^ XOR(both true or both false return false)    ~ NOT   << (add 0 right most)    >>
    
    --Math
    sys.maxsize   # max integer     -sys.maxsize   min integer
    float("Inf")   -float("Inf")    # max min float
    abs(-10)   # 10
    pow(3, 2)   3 ** 2  # 3^2
    
    --Binary and nary 
    bin(149)  # 0b10010101 string  binary     oct(149)  #0o225  8     hex(149) # 0x95 16     int() # back to decimal
    -7 in binary: 7 in binary ob0000 0111,    reverse ~  ob1111 1000,  then +1  ob1111 1001     ~n+1 is -7 
    binary(negative) to decimal: ob1111 1001,   -1  ob1111 1000, then reverse ^ ob0000 0111  is 7
    print(~7+1)  # -7          n << 3    # 7 * 2^3               n >> 2   # 7 // 4
    print(~7 + 1)  # -7           print(7 << 3)  # 7 * 2^3         print(7 >> 2)  # 7 // 4
    
    token = uuid.uuid4().hex # Generate a random UUID. hexadecimal uuid
    
    --Condition
    if boolean condition:       
        pass
    elif x < -1:
        pass
    else:
        pass
    can be simplify to:    result = 4 if age > 5 else age
    
    
    --Loop
    while condition:
        update condition so it can end after some runs
        continue  # skip later code in loop and continue to next iteration
        break  # end loop 
    else:
        pass   # run one time after while loop end
        
    for i in range(5):  # [0, 5)      range(1, 5):  # [1, 5)    range(1, 10, 3)  # [1, 10) step size 3
        pass    # i can only be accessed to inside loop
    else:
        print(i)  # run one time after for loop end, 4
        
    for loop only for certain iterations, while loop can handle uncertain iterations
        # range(0) won't cause exception
        
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    list(enumerate(seasons))    #[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]     
    for i, element in enumerate(seasons):
        print(i, element)     # 0 Spring ,...
    
    
    list, set, dict are equal o1 == o2 if has same values inside, but have different memory location, unless
    o1 = o2 assignment. tuple always same location if value same.
        
    --List
    mutable array able to store different data type entries
    Create / Copy
    list1 = []  list1 = [1, '2', True]    list2 = list(str)  # create char list from string
    list1 = list()
    list(range(3,5))  # [3,4]       [i * 2 for i in [1, 2, 3, 4] if i % 2 == 0 and i >= 0]  # [4, 8]      
    [i for i in 'hi']  # ['h', 'i' ]   
    [w.lower() if w.startswith('h') else w.upper() for w in list1]  list1 item start with h then lower otherwise upper 
    [(x,y) for x in range(2) for y in range(3)]  return all combination of (x,y)
    list2 = list1.copy()  or  list2 = list1[:] # deep copy, change in new list won't affect original, 
        # if list2 = list1, when list1 changed, it will change list2 as well since they have same memory location
    initialize list: [0] * 5   [ 0 for i in range(5)]     [[0] * 5 for row in range(3)]  
        [[0 for i in range(10)] for j in range(10)]
    
    Read
    print(list1[0], list1[:1], list1[::-2])  # [True, 1]       [:1] [0,1),  from last item to front step 2
    len(list1) #3    
    list1.index(1)  # 0  exception if not found    list1.index(1,0,2)  find 1 in index between [ 0,2)
    list1.count(1)  # 1  number of appearance of value
    Add   mutate list
    list1 = list1 * 2  # [1, '2', True, 1, '2', True]     # list1 *= 2
    list1 + [100]    # [1, '2', True, 100]  # not mutate original list
    list1.append(100)  # [1, '2', True, 100]   
    list1.extend([100, 200])  # [1, '2', True, 100, 200]   # iterable input parameter
    list1.insert(-1, False)   # [1, '2', True, False]  # insert item at index, shift one to the right after index 
    Update
    list1[1] = 2    # don't have return  update list existing index with value, otherwise IndexError
    Delete   mutate list
    list1.pop()    # True  default index is -1, can pop empty list(no error)
    list1.pop(1)   # '2'   exception if larger than list length
    list1.remove(True)  # return None [1,'2'] Removes first occurrence of item or raises ValueError. 
        # if True in list1:  list1.remove(True)
    [1,2,3].clear()  # return None  []  removes all items
    del list1[0:1]    #  return None  [True]  
    Sort
    list1.sort()   # return None  mutate original list  small-> large
    list1.sort(reverse=True)    # return None  mutate original list  large -> small
    list1.reverse()   # return None   mutates list to [True, '2', 1]
    li = sorted(list1)  # return new sorted array
    reversed(li)  # return iterable object   for i in reversed(list1): print(i)   [True, '2', 1]
    Aggregate functions
    min(list1)   max(list1)  sum(list1)
    
    
    --Tuple
    similar to list, immutable, keep order, less flexible, better performance
    my_tuple = (1,2,3,'x')    # (2,) for only 1 element
    print(3 in my_tuple)    #True
    print(my_tuple[1:2])   # (2,)  tuple has one element has , at end to distinct ()
    a, b,*c = my_tuple   # 1   2   [3,'x']
    my_tuple.count('x')  #1      my_tuple.index(1)   #0     len(my_tuple)   #4
    t = tuple(list1)  # convert list1 to tuple
    from collections import namedtuple  # hashable named tuple variation   
    can use tuple.key to get value
    
    
    --Dictionary
    Also known as map or hashtable (key value pair), retain order of insertion when iterating keys after Python 3.7
    dictionary key need to be immutable: int, float, boolean, string, None, tuple
    same dictionary key, the value will be override    use hash O(1) operations
    Create
    my_dict = {'name': 'Andrei Neagoie', 'age': 30, 18: False}    dict1 = {}   dict2 = dict1.copy()
    dict1 = dict()
    dict2 = dict([('name','Harry'),('age',[10,1])])
    dict2 = eval('{"name":"Harry", "age":10}')
    dict2 = json.loads('{"name":"Harry", "age":10}')  # must double quote
    dict2 = dict.fromkeys(['name','age'], 1)   # create dictionary with keys and same default value   
    {k:v for k, v in dict1.items()}     {k: v for k, v in [('one',1),('two',2)]}
    Read
    dict1['name']       # Andrei Neagoie   keyError if key not exist
    dict1.get('age')                   # 30 --> Returns None if key does not exist.
    dict1.get('ages', 0 )              # 0 --> Returns default 0 (2nd param) if key is not found   
    len(my_dict)        # 3  
    'name' in my_dict   #True
    list(my_dict.keys())                 # ['name', 'age', 18]
    list(my_dict.values())               # ['Andrei Neagoie', 30, False]
    list(my_dict.items())                # [('name', 'Andrei Neagoie'), ('age', 30), (18, False)]  list of tuple    
    {k: v for k, v in my_dict.items() if k == 'age' or k == 'name'}  # {'name': 'Andrei', 'age': 32} Filter dict by keys
    Update/Insert   mutate dict
    dict1['fruit'] = 'apple'  # Add key-value {'fruit': 'apple'}
    dict1['fruit'] = 'pear'  # update key-value if key already exist {'fruit': 'pear'}
    dict1.setdefault('fruit','pear')  # only add key-value, can't update
    dict1.update({'fruit':'peach'})    #create new pair if not exist   
    Remove  mutate dict
    del my_dict['name']   # delete key name, keyError if key not exist
    name = my_dict.pop('name',None)  # delete key name, return value of second param. keyError if not specify
    (k, v) = my_dict.popitem()   # delete last item, if no item exist throw keyError 
    dict1.clear()  # clear whole dictionary, empty dictionary remained
    del dict1  # delete the dictionary structure as well
    dict can convert to list tuple set but only keep keys           
    d_order = dict(sorted(d.items(),key=lambda x:x[1],reverse=True))   # sort dict values descending
    
    --Set
    Unordered collection of unique item, no order, use hash   use hash O(1) operations
    Create
    set1 = set()  set1, set2 = {1,2,3},{3,4,5}    set(new_list)    new_set = set1.copy()  
    {x-1 for x in list1 if x > 5}
    Read
    1 in set1   #True
    Insert
    set1.add(1)  # return None {1}  won't add additional duplicate item
    Update
    set1.update(set2)  # add set 2 to add 1, set 2 unchanged
    set3 = set1.union(set2)               # {1,2,3,4,5}    set1 | set2
    set4 = set1.intersection(set2)        # {3}    set1 & set2
    set5 = set1.difference(set2)          # {1, 2}  unique item in set1
    set1.difference_update(set2)   # modify set1 to unique item only  {1, 2}
    set1.issubset(set2)                   # False  child set
    set1.issuperset(set2)                 # False  parent ser
    set1.isdisjoint(set2)                 # False --> return True if two sets have a null intersection.
    Delete
    my_set.remove(1)      # return {1} --> Raises KeyError if element not found
    my_set.discard(1)     # {1} --> Doesn't raise an error if element not found
    my_set.clear()          # {}
    set1.pop()   # random delete one item
    del set1   # delete set1 object  
    
    collections.OrderedDict preserves the order in which the keys are inserted. only difference with dict
    
    --Function
    modularize repeated content
    def function_name([para1,...]):
        pass
    function_name([para1_value,...])  # call function
    def function_name([para1, para2='default_value']):  # Default value can't use [] or {}, a new list is created once 
        when the function is defined, and the same list is used in each successive call. Python’s default arguments are 
        evaluated once when the function is defined, not each time the function is called.
        when calling function with default value, the parameter with default value can be omitted
    function_name(para1_value)   or function_name(para1=para1_value)
    function_name([para1,...],*args):  # front parameters are required, *args pack additional parameters into a tuple
        args[1] # access the parameter in tuple with index 1   *args  can be 0 parameter as well
    function_name(para1_value, 1, 2)    # args[1] will be 2
    function_name(*[1,2,3])    # *[1,2,3] unpack list to individual parameters, work for tuple, set as well
    function_name2([para1,...],**kwargs):  # front parameters are required, *kwargs pack additional into dictionary
        kwargs['name']   # 'JK'
    function_name2(1,name='JK',date='1991-10')    # will call need para=val, here para no need ''
    function_name2(1,**{'name':'JK','date':'1991-10'})
    function_name3(*args, **kwargs): 
        return kwargs['name']   # function return value,  or just use return to step out the function. 
        return kwargs['name']   # can have multiple return statement under various if condition, can't return multiple 
                                # value in one return statement
    name = function_name3(1, 2,**{'name':'JK','date':'1991-10'})  # name is 'JK' for first return  
    item = function_name3(1, 2,**{'name':'JK','date':'1991-10'})  # item: ('JK','1991-10')  put return items in a tuple
    name,date = function_name3(1, 2,**{'name':'JK','date':'1991-10'})  # name is 'JK', date is '1991-10' for 2nd return
        # **{'name':'JK','date':'1991-10'}   # name='JK', date='1991-10'  for parameters
    if immutable variable var, inside function, add: global var    var = 7  to change the global outside variable values
    
    def funcName(digits: str) -> List[str]:  # declare input type and return type
    
    a = 6; b = []  
    def sum():
        global a
        a = 7
        b.append(1)
    print(a,b)  # 7 [1]   # immutable data will not change if no global declaration, mutable data will always change
    inside function add document comment   ''':param name: xxx   :return: xxx '''
    help(function_name)   # print out function information,  hover on function name will show document comments
    
    def outer():
        a = 100    # outer function can't access inner function variable, since it's function is destroyed after finish
        def inner():
            nonlocal a    # immutable data type a cannot be modified before nonlocal declaration
            a = 200  # inner function can't modify outer function variable, it only create a new inner variable when
            print(a+10)     # have the same name as outer variable, unless add nonlocal var declaration   
                         # when variable is used, find declaration with order: inner -> outer -> global -> builtin    
        print(inner)  #return location info
        info = locals()  # info will have local variable and value, inner function location information
        # inner()  # run inner function
        return inner  # return function
    r = outer()  # return inner function memory location
    r()  # run inner function   or  outer()()
    
    info = globals()  return the global variable (system and user-defined) dictionary 
    
    decorator pattern
    def d(func):      
        def wr(para):      # def wr(*args,**kwargs):  to cover all input cases
                 # func(*args,**kwargs) 
            print(1)       # add extra logic here
            return func(para)
        return wr   #
    @d      # <- ->  equivalent to:  f = d(f), used for add additional logic while keep original function name and call
    def f(para):
        print(2)  
        return 'x'
    r = f(para_val)   # 2  1  # first execute outer function of d then go into inner function
                  # equivalent to call wrapper function
                  
    decorator function can have input parameters as well need extra layer of outer function
        functions inside function can't be directly called
    def with_param(para2):
        def decor(func):
            @wraps(func)    # for f.__name__, return function name (f) instead of decorating function name (decor)
            def d(*args. **kwargs):  # def d(para1) if known parameter
                print(para2)   # add extra logic here
                return func(*args. **kwargs)
            return d
        return decor
        
    @with_param(para2_val)
    def f(para1_val):
        return para1_val+para2_val
    
    
    --Recursion
    must have exit conditions, each step closer to exit
    def p(start,total):       def p(start):                                         def p(v):           
        if start == 0:                                                                 if v == 0:
            return total                                                                   return
        return p(start-1, total+start)   # tail recursion, save space                  p(v-1)
        return start+p(start-1)     # keep older call stack                                
       
                                            
    --Anonymous function (lambda)
    func = lambda arg1, arg2: expression (ex: arg1+arg2)
    print(func(1,2))  # 3
    def func1(a, f):
        return f(a)
    func1(3, lambda x: x+2)  
    x = lambda a : a + 10;    print(x(5))   
    list1 = [('sam',35),('tom',19), ('tony',20)] 
    print(max(list1, key=lambda x: x[1]))   # ('sam',35)   key is a function x is each tuple in list1                              
    max, min, sorted all have key function
    print(list(filter(lambda x:x[1]>19, list1)))  # [('sam', 35), ('tony', 20)] filter out data not match condition
    print(list(map(lambda x:x[0].title(), list1)))  # ['Sam', 'Tom', 'Tony']  retrieve part of data
    from functools import reduce
    print(reduce(lambda x, y: x+y, [1,2,3,4,5]))  # sum of array, or concat strings
    r = zip([1,2,3], {'one','two','three'});  print(set(r))  # {(1, 'one'), (2, 'two'), (3, 'three')}
    print(list(r))  # [(1, 'one'), (2, 'two'), (3, 'three')]  
    print(dict(r))  # {1:'one',2:'two', 3:'three'}
    # The zip() function returns an iterator of tuples (built by each iterable input).
    
    
    --File operation
    $touch test.txt   #create new file       $ test.txt  #open file
    Opens a file and returns a corresponding file object.
    file = open('<path>', mode='r', encoding=None)
    Modes    default rt
    'r' - Read (default).
    'w' - Write (truncate, or create new file).
    'x' - Write or fail if the file already exists.
    'a' - Append.
    'w+' - Read and write (override from beginning).
    'r+' - Read and write from the start.
    'a+' - Read and write from the end.
    't' - Text mode (default).
    'b' - Binary mode.  (audio, video, image...)
    
    my_file = open('test.txt')    open('path/test.txt', mode = 'a', buffering, encoding)  return a steam object
        # FileNotFoundError if path not exist
    print(my_file.readable())    # return boolean whether stream is readable
    print(my_file.read()) #read whole file
    my_file.read(1024)   # read next 1024 bit in file
    my_file.seek(0)  # reposition cursor to start so can read again
    my_file.readline()  #read one line and move cursor down 
    my_file.readlines()  # return list of lines
    my_file.write(string)   # in 'w' mode will clear content then write
    my_file.writelines(Iterable)   # need add  \n in string to switch new line
    my_file.close()  # release resources
    or use with open() as stream to auto release resources   
    try:
        with open('path/test.txt', mode = 'a') as my_file:    
            # windows       open(r'C:\path\test.txt', mode = 'a')   open('C:\\path\\test.txt')
            #  ./ from current folder      #  ../ back one folder
            my_file.write("Hello World")
            print(my_file.readlines())
        except FileNotFoundError, IOError as err:
            print('file does not exist')
        
    with open('data.json') as f:
        dic = json.load(f)   # load string or bytes into dict
        print(dic['age'])
    
    create json object 
        import json
        data = json.dumps({'a': 123, 'b': 456}, separators=[',', ':'], ensure_ascii=False, indent=4, sort_keys=True)
                .encode('utf-8')  
            # separators use ',' instead of default ', ',  use ':' instead of default ': 'remove space
            # ensure_ascii=False: prevent convert utf8 to ascii
            # indent=4: add indent for easy reading
            # json.dumps() return string   encode change string to machine language(bytes), decode change machine 
                language to string
        if object is not JSON serializable, write a function to return necessary attribute and value as dictionary
        
        json.loads(data)  # load bytes or string to dict 
        
    import os   
    os.path.isabs(string)   # return bool check path is absolute path
    os.path.abspath('python_notes.py')  # return file name's absolute path
    os.path.abspath(__file__)  # return file name's absolute path
    os.path.dirname(__file__)  # return string of current file's absolute directory
        os.getcwd()  # return string of current file's absolute directory
    os.path.split(path)   # return tuple of directory and file name  ('C:\\path\\','a.txt')
    os.path.splitext(path)  # return tuple of file extension and directory+file name  ('C:\\path\\a','.txt')
    os.path.getsize(path)   # return file size in bytes
    os.path.join(os.getcwd(),'path','a.txt')     # return path with path\\a.txt inside current directory
    os.path.abspath(os.path.join(os.getcwd(), '../..', 'media')) # return path ../media
    os.path.listdir(path)   # return a list containing all the file and directory under path or current path if no input
    os.path.exists(path)   # return bool   check path exist or not
    os.path.isfile(path)   # check path is file or not
    os.path.isdir(path)   # check path is directory or not
    os.mkdir(path)   # no return  raise FileExistError if path already exist
    os.rmdir(path)   #   remove empty directory   otherwise raise OSError
    os.remove(path)   # remove file 
    os.chdir(path)   # switch current working directory to path
    os.getpid()      # get process id

    --Exceptions
    parent class: BaseException
    try:
        pass
    except :      # can use multiple except  
        print('check input')
    except ValueError as err:   # except block can raise exception as well
        print(err)   
        raise Exception('wrong input')    # need another try catch block outside to handle raised exception
    [finally:    # always execute no matter exception block executed or not, even execute when try block has return 
        pass]        # statement, but if finally has return as well, then it will override try block's return
    [else:      # only execute if no exceptions, can't use together with finally 
        pass]
        
    
    --Generator
    from a list, each iteration only retrieve a few data to save memory cost if the list is large
    1. g = (x*3 for x in range(20))  # generator type
       g.__next__()    or   next(g)    # to retrieve next value 
       # can not generate more item than length of list, raise exception, use try, catch block
    2. use yield function
        def func():        def func(para):
            n = 0
            while True:    while n < para: 
                print(n)              
                yield n   # pause the function until next call of function      
                    # temp = yield n   # receive input parameter from outside (send function param)
                print(temp)  # temp = 3
                n += 1   
        [return 'can't generate more']   # return message when loop end with exception
        g = func()    # retrieve generator    g=func(para_value)
        g.__next__()   next(g)    # to retrieve next value after yield (n)
        # raise exception when while end, use try, catch block
        sending input parameter
        r0 = g.send(None)   # r0=0    # first time call must send None or next(g)
        r1 = g.send(3)   # r0=1   temp=3    temp=3 at  temp = yield n
        
        Coroutines: sub thread, one thread can have multiple Coroutines running concurrently
        def task1(n):
            for i in range(n):
                print(n)
                yield None
        g1,g2= task1(10), task2(5)
        while True:
            try:
                next(g1)
                next(g2)
            except StopIteration:
                break    # wont have error message 
                
                   
    --Iterable and Iterator
    iterator can save the item index while traversing the items of object. Can only traverse forward till end.
    iterator object it, can use next(it) to repeatedly access next item in the iterator object.
    iterator is iterable, but iterable is not necessary iterator
    generator is an iterator, list is iterable, but not iterator
    user:  iter(Iterable)     ex: it=iter([1,2,3]); next(it)   to change iterable list into iterator
    isinstance(var, Iterable)   # return bool   check whether variable data type is iterable or child class of iterable 
    
    --OOP object oriented programming
    class name upper camel case, default extend from parent: object    class CellPhone:  class ClassName(object):
    multiple inheritance     class CellPhone(Commuter, Electronic):  
    import inspect     print(inspect.getmro(CellPhone))    or print(CellPhone.__mro__)   # get inheritance order
        # Method Resolution Order
    # inheritance order CellPhone->Commuter->Electronic->object   python1 3: bfs    python1 2: dfs preorder
    class CellPhone(Commuter):    # non specific parent class extend from object, here extend Commuter class
                                  # child class have all parent class attributes, need to override if necessary
                                  # extension eliminate duplicate code  
                                  # child will override parent same name, # parameter method, but once override with
                                  # different # parameter, child can no longer access parent's same name method
        def __new__(cls, *args, **kwargs):   # declare a new space for the object, usually not needed, extend parent's
            return object.__new__(cls)     # return new space location then pass into __init__'s self
        def __init__(self,price=0):   # init (constructor) is ran once instance is initialized  
            self.price = price        # use (self, **kwargs for overload)
            self.brand = 'Huawei'     # default instance variable value
            self.case = Case()        # instance variable can be different for different CellPhone object, first search 
            super().__init__(name)    # instance whether variable is defined, then search in class 
            # super(CellPhone, self)  # instance variable can be other class instance object  (has relationship)
                                      # need call parent's init method, need same number parameter
                                      # super(CellPhone, self) same as super(), but add self instance type check
        model=''    # class variable, shared among all instance
        __pin=''    # private variable, can't access/ modified outside class with CellPhone.__pin
                    # variable was rename to _CellPhone__pin   still able to access ph._CellPhone__pin, not recommended
        def start_phone(self):    # instance method(function), self is the current instance (required if need access 
            print(self.brand)     # other access instance variable or function defined in the class)
                                  # need first initialize instance, then can use instance method
        def close_phone(self, name):   # method with input parameter
            self.startphone()     # inside method, use other non-class method must use self
        @classmethod   #decorator  class method usually used for define actions taken before instance created
        def destroy(cls):         # class, don't depend on instance. every instance have the same class method
            cls.__pin = '1'       # don't need initialize instance to use class method, but ph.destroy work as well
                                  # inside class method, can only use / update class variable and class method 
        @staticmethod
        def create(name):         # static method, no self/cls, can only use / update class variable and class method 
            Person.model='Xiaomi' # used for define actions taken before instance created
        def __call__(self,para):  # needed when run the class instance like method:  ph(1)
        def __del__(self):        # usually don't need to write, use parent's object.__del__. executed when there is no       
                                  # reference to a class object or at the end of execution 
        def __str__(self):        # return string, used for print(ph) debugging with specific info  without __str__,  
            return self.brand     # will print memory location    
        # __xxx__ function will be trigged automatically, no need direct calling
        def setPin(self, pin):    # oop encapsulation, private variable, public getter and setter 
            self.__pin = pin      # can have additional logic in setter function to avoid unwanted input
        def getPin(self):
            return self.__pin
        # alternative approach of getter and setter, use property decorator  
        @property        # read only if no setter, still a function
        def pin(self):            # must first have getter then setter, instance now can use ph.pin to get and set
             return self.__pin
        @pin.setter            
        def pin(self,pin):
            self.__pin = pin
                     
    ph = CellPhone()   # use class to initialize object
    ph.brand='xiaomi'  # can add extra instance variable (feature) to instance after creation
    ph.model='xiaomi'  # change in value of variable with the same name won't change the variable value inside class 
    CellPhone.model = 'huawei'  # this will change the variable value defined inside the class
    ph.close_phone('Harry')
    CellPhone.destroy()    # no need initialize instance for class method
    CellPhone.create('mine')    # no need initialize instance for static method
    ph(para_val)  # when run the instance like method, this will trigger class' __call__ method
    ph2 = ph  # create a new reference point to the memory, so change in ph2 will change ph as well
    del ph2   # delete the location reference of ph2 to ph
              # if there is no reference to a class object or at the end of execution, __del__ will be used to do 
              # garbage collection
    ph.setPin(12)
    print(ph.getPin())  # 12
    dir(ph)  or ph.__dir__()  # return all the attributes of the object and parent (methods and non-private variables)
    review.pin=10  # with property decorator
    print(review.pin)  # 10
    hasattr(ph, 'brand')  # check whether ph object has attribute brand
    
    for field in self._meta.fields   # return list of all self defined attribute
        name = field.attname
        value = getattr(self, name)
    
    polymorphism： same method base on input type (isinstance(var, type)), run different code block
    
    class Singleton: 
        __instance = None    # when initialize object, always return the same one
        def __new__(cls):   # return __instance if already exist, otherwise allocate new memory location
            if not cls.__instance:
                cls.__instance = object.__new__(cls)
            return cls.__instance
    
    
    --Module
    each .py file is a module (ex: os, builtins), each file contains similar functions or class. this improve code 
    reuse via import module. builtins is default imported
    import module_name   (.py file  without .py)   ex. import basic_numpy
    from module_name import var/method/class    # can direct use var/method/class  no need add module ahead
    from basic_numpy import var_a, var_b, CellPhone     
    from basic_numpy import *   in the module can use __all__=[var,method,class]  to specify items import * can access
    basic_numpy.method()  basic_numpy.variable    basic_numpy.class_name
    import will load everything from the module, if anything don't want to be run when imported, 
        add code inside if __name__ =='__main__':      __name__ will change from __main__ to module name if was imported
    directory and packages
    directory hold non python file, package hold python file
    directory will become a package after add an empty __init__.py file (when module imported, execute automatically)
    package name can only use number alphabet and _
    from package_name import module_name
    from package_name.module_name import class_name
    from .module_name import class_name   # imported module and current module have same parent directory
    __init__.py file can include some initialize common variable, method, class. so when package imported, those things 
        like can be accessed through package_name.methods
        need include __all__=[var,method,class]  to use from package import *
    
    when recursive import occurs with two python file, maybe move the import statement just before actual usage will 
        resolve the issue, otherwise need redo architecture  
    
    print(sys.path)  # return list of paths which represent the search order of import module
    system module
    import sys    sys.getrefcount(var)   # return the number of variable using the reference var
    sys.version  # python interpreter version
    sys.argv   # argument parsed for running the module
    
    
    --Time
    import time
    time.time()    # return float point number
    time.ctime(time.time())      # return string of time in format of Mon Sep  6 23:55:30 2021
    t = time.localtime(time.time())  # return named tuple of time: time.struct_time(tm_year=2021, tm_mon=9, tm_mday=6,
                                  # tm_hour=23, tm_min=57, tm_sec=39, tm_wday=0, tm_yday=249, tm_isdst=1)
                                  # t.tm_hour  # 23    function default input is current, so time.localtime() works
    tt= time.mktime(t)           # change tuple back to float
    time.strftime('%Y-%m-%d %H:%M:%S')    # return string of time with specified format of current time
    time.strptime('2019/06/20','%Y/%m/%d')  # return named tuple of time 
    time.sleep(n)   # sleep for n seconds
    import datetime
    birthday = datetime.date(2019,6,20)
        # today = datetime.date.today() # 2021-09-07     
    print(birthday.day)  # 20         print(datetime.date.ctime(birthday))  # Thu Jun 20 00:00:00 2019
    b2 = datetime.datetime(2019,6,20,10,30)        print(datetime.datetime.ctime(birthday))  # Thu Jun 20 10:30:00 2019
    now, delta = datetime.datetime.now(), datetime.timedelta(hours=2)  #(weeks=3,days=2)
    datetime.strptime('2022-10-31 16:55:00', '%Y-%m-%d %H:%M:%S'))
    print(now, now - delta)   # 2021-09-07 00:19:10.488159  2021-09-06 22:19:10.488159
    
    x = birthday - today  # datetime.timedelta
    x.years  # return difference years
    
    --Random
    import random
    random.random()  # Return the next random float in the range [0.0, 1.0)
    random.randrange(start,end,step)  # random choose a number from [start, end) with step default 1 
    random.randint(start,end)  # random integer [start, end] 
    random.choice([1,2,3,4,5])   # random choose from a sequence
        random.choice('12345')   
    random.sample([1,2,3,4,5], 3)   # random pick 3 from list return as list
    random.shuffle(li)   # shuffle the sequence in random order for an object inplace
    random.seed(0)    # set random seed so random will get same value
    
    
    --Hashlib
    import hashlib
    md5, sha224 those are hash functions, generate a unique value, if modified the original data, this value will change 
        as well. the unique value also known as electronic signature
        don't save password directory into database, save the string generated by hash function
    md5 = hashlib.md5()   # initialize md5 object
    md5 = hashlib.md5("hello".encode('utf-8'))      # md5 sha224  sha256 not invertible        base64 is invertible
    md5.update("world".encode('utf-8'))  # md5 is now calculating hash for: hello world, return None, must encode
    print(md5.hexdigest())    #  5d41402abc4b2a76b9719d911017c592     save encoded password in database
    sha224 = hashlib.sha224("hello".encode('utf-8')) 
    print(sha224.hexdigest())   # convert to hexadecimal ea09ae9cc6768c50fcee903ed054556e5bfc8347907f12598aa24193
    
    
    --Regular Expression
    import re
    '''
        . : any character except \n     ^ : search pattern from start     $ : search pattern till end    | : or 
        [] : one character from range of option      \s : space      \d : digit    \w : [0-9a-zA-Z_]   \D : not digit   
        \b: Matches at beginning or end empty string, or boundary between a \w and a \W character  (163|qq) : 163 or qq
        counters  
        * : >=0    + : >= 1    ? : 0,1    {m} : m character   {m,} : >= m characters   {m,n} : [m,n] characters
        counters are greedy will return the longest matched string after one match  , add ? to make it not greedy
              re.match(r'abc(\d+)','abc123de')  # abc123 ,  re.match(r'abc(\d+?)','abc123de')   # abc1
        assign name to part of pattern:  ?P<name>  to assign later pattern in () with name, ?P=name to match the pattern
        msg = '<html><h1>abc</h1></html>'   re.match(r'^<(?P<name1>\w+)><(?P<name2>\w+)>(.+)</?P=name2></?P=name1>'$)
             or use number \1 to match first group()    re.match(r'^<(?P<name1>\w+)><(?P<name2>\w+)>(.+)</\2></\1>'$)
    '''
    msg = 'caichenghao@gmail.com';   
    pattern = re.compile('^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$')
    result = pattern.match(msg)      # <re.Match object; span=(0, 21), match='caichenghao@gmail.com'>
    result = re.match(pattern,msg)   # return None if no match, match the pattern from beginning of the string,  
                   # return <re.Match object; span=(0, 10), match='123&abc456'>  span shows the index of matching
    re.match("([0-9]*)&([a-z]*)([0-9]*)", msg)  
    result = re.search(pattern, msg) # search can match the pattern not just from beginning, pattern add ^xxx$ to ensure 
                                     # search from beginning and till the end, stop once one result is found
                                     # return re.Match object
    result.span                      # (0, 21)    return matched position, for re.Match object
    result = re.findall(pattern, msg)   # return a list of all matched results                
    result.group()                   # return the string matching the pattern, default parameter is 0, return all the 
                                        # matched string with pattern
    result.group(1)                  # return string match the first () in pattern, index start with 1
    m = re.match(r"(\d+)\.(\d+)", "4.16")     # ('4', '16')Return a tuple containing all the subgroups of the match, 
                                     # r will suppress interpretation of string \t,\b
    result.groups()                  # return a tuple with all matched () pattern, same as (result.group(1), 
                                        # result.group(2),result.group(3),...)
    result = re.sub(r'\d+','90','{java:99,python:100}')  # replace, match 1st pattern with 3rd string, and replace 
                                     # matched item with 2nd parameter (can be a function which return string)
    result = re.split(r'[,:]','aa:b,c')  # ['aa','b','c']  return a list split the string by matched pattern 
        
    
    --MultiProcess  (mainly used for large calculation)
    concurrency: multiple tasks don't own the cpu entirely, cpu handle one task for a short period, and switch to next 
        task, then next...
    parallel: each task have dedicate cpu to handle the task 
    process: execution unit where a program runs. basic block of system resources management
        high stability, one failing won't affect others. demand more limited resources
    
    Linux use fork() function in os module to start new process    pid=os.fork() (<0 failed, 0 child process, else pid)
    windows use multiprocessing module process class (__init__(), start(), terminate(), join()) 
    Windows 
    from multiprocessing import Process
    n = 0   # global variable when accessed by a process will create own copy, not shared among other process
    def task1(n):
        while True:
            time.sleep(n)
            global n   
            n+=1
        return n   # needed if there is a callback function
    def call_back(n):
        print(n)
    
    p = Process(target=task1, name='job 1', args=(1,))   # child process, args can be tuple or list, name optional
    p.start()                             # start and run process
    print(p.name)  # job 1
    p.run()                               # run process
    p.join()                              # main process not stop until child process finished
    p.terminate()                         # stop process
    
    self-defined process
    class MyProcess(Process):
        def __init__(self, name):    # override (optional)
            super(MyProcess,self).__init__()
            self.name=name
        def run(self):     # override
            time.sleep(1)
    p = MyProcess('job')
    
    process pool: define number of max process n, if current process less than n, create a process, otherwise wait until
        one process finish, and less than n process. pool will reuse pid when task finish. pool dies if main process die 
    from multiprocessing import Pool
    pool = Pool(5)   # max 5 processes 
    for t in tasks:                                               # call back function execute after task finished          
        pool.apply_async(task1, args=(t,), callback=call_back)    # async mode won't wait task finish to add next 
        pool.apply(task1, args=(t,))                        # sync mode won't add new process until first task finished
    pool.close();                                                 
    pool.join()   # put pool inside call stack, main process won't stop until pool finish
    
    communication among processes  
    from multiprocessing import Queue
    q = Queue(5)    # max items inside the queue is 5
    if not q.full()     # check queue is full or not  
        q.put('A',timeout=3)      # add item into queue, if larger than max, the process will wait until queue have  
                                  # empty space, can add timeout to raise error if wait more than 3 seconds
    if not q.empty():     # check queue is empty or not  
        print(q.get(timeout=3))    # will wait until have item in queue, can add timeout to raise error 
    
    def sender(q):
        time.sleep(0.5)
        q.put('message')
    def receiver(q):
        while True:
            try:
                msg = q.get(timeout=3)
            except:
                break
    q = Queue(5)    # max items inside the queue is 5
    p1, p2 = Process(target=sender, args=(q,)), Process(target=receiver, args=(q,), daemon=True) #daemon task
    p1.start();  p2.start();  p1.join(); p2.join(); print('end')
    
    
    --Multi Threading  (mainly used for time consuming(IO) tasks)
    Thread is an execution unit that is part of a process. thread is rely on its process
    threads runs concurrently inside process 
    thread state: new -> ready -> running -> sleep -> ready -> running -> end
    import threading
    ticket = 1000  # default add apply GIL (global interpreter lock) unless large computation
                    # can cause problem when large number (release GIL when large computation), ticket -=1 is two step 
                    # process, another thread can enter when second assign step haven't finished
    t = threading.Thread(target=sender, name='',args=(q,))   # thread object daemon task
    t.start()    # start and run thread, thread can access same global variable
    t.join()     # put thread inside call stack, main process won't stop until thread finish
    
    # to avoid inaccuracy caused by multiple thread using shared resources, lock is introduced
    lock = threading.Lock()
    def seller():
        lock.acquire(timeout=3)   # wait until the other lock holder release lock, timeout avoid deadlock
        ticket -= 1
        time.sleep(0.1)
        lock.release()
    
    class MyThread(Thread):
        def __init__(self, name):    # override (optional)
        def run(self):   # override
    # to avoid deadlock, either redo architecture, or add timeout in lock.acquire()
    
    import queue
    def producer(q):
        for i in range(10):
            q.put('item')
            time.sleep(0.5)
        q.put(None)
        q.task_done()     # Queue.task_done notice the queue, when all task is done, queue.join will stop blocking
        
    def receiver(q):
        while True:
            item = q.get()
            if item is None:
                break
            time.sleep(1)
        q.task_done()     
    q = queue.Queue(10)
    tp,tc = threading.Thread(target=producer,args=(q,)), threading.Thread(target=consumer,args=(q,))  
    tp.start(); tc.start(); tp.join(); tc.join(); print('end')
    
    
    --Coroutines  (mainly used for time consuming(IO, web) tasks, better than thread in IO)
    Coroutines allows to switch task, when the current task (non calculation related) takes longer time 
    use yield or use greenlet (need specify switch task)  or use gevent (automatically switch task)
    from greenlet import greenlet
    import gevent 
    yield:                           greenlet:                                gevent:  
    def task1(n):                                                             # monkey.patch_all() ahead of tasks
        for i in range(n):                                                    # monkey will change the native time module 
            print(n)
            yeild                    # gb.switch(n_val)                       # no need for gevent
                # yield n  # return n   # manual switch to task2
            time.sleep(0.5)
             
    g1,g2 = task1(10), task2(5)      # ga = greenlet(task1)                   # g1 = gevent.swpan(task1, n_val) 
    while True:                      # gb = greenlet(task2)                   # g2 = gevent.swpan(task2, n_val) 
        try:                         # ga.switch(n_val)                       # g1.join()    # gevent.joinall(g1, g2)
            next(g1)                                                          # g2.join()
            next(g2)  # raise error if no next item                                     
            # x = next(g1)  # will get the yield return value 
        except:
            pass
    
    asyncio  (python official implementation)
        import asyncio
        async def task1(n)
            for i in range(n): 
                print(n)
                await asyncio.sleep(1)
            return i
    t1, t2 = task1(1), task1(2)   # coroutine object
    tasks =[asyncio.ensure_future(t1), asyncio.ensure_future(t2)]   
    loop = asyncio.get_event_loop()   # coroutines dispatcher
    loop.run_until_complete(asyncio.wait(tasks))   # epoll 
    print(tasks[0].result)  # 1
                
    context: state stored and read during cpu switching tasks every several milliseconds
    process: ~MB, communication: socket, pipe, file, shared memory, UDS. context switching a bit slower, not flexible, 
        controlled by operating system
    thread: ~kb, communication: direct transfer information. context switching a bit faster, not flexible, controlled by 
        python interpreter. wasting cpu resource if process / thread blocking
    Coroutines: <1k, communication: same as thread.  context switching fast and flexible (controlled by programmer)
    high performance context switching: blocked task take no cpu time, only switch to the blocking task after receive
        an I/O event notification (listened by OS interface: select, poll, epoll(add event in readied queue instead of 
        looping listening to event))
    event driven: take action after receiving an event. Nginx (40000+ RPS), (110000+ Redis), (5000+ Tornado)
    compare to Django 500 RPS
    
    multi-process(used because of global interpreter lock, limit to 1 task per process at any time send to cpu) 
        + mult-coroutine implementation for performance, allowing multiprocessors handling multiple tasks for 
        multi-process 
    
    --Request
    import requests  
    import urllib.request
    response = requests.get(url) # return response object
    content = response.text   # string of webpage  
    response.json() #json object
    response.content   # binary data of webpage
    
    # response = urllib.request.urlopen(url)
    # content = response.read()   # binary data of webpage
    with open('aa.jpg','wb') as ws:
        ws.write(requests.get(path).content)
    
    
    --Testing
    assert a == 5  # raise AssertionError if not equal
    
    
    --Others 
    id(variable)  # get the readable memory location (integer) of the variable stored
    
    pip install -U pip   # update pip  
    pip install pymysql==0.9.2   # install specific version
    pip freeze   # show all dependencies 
    pip freeze > requirements.txt
    pip install -r requirements.txt
    
    # logging config    
    LOGGING_CONFIG = { 
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': { 
            'standard': { 
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(funcName)s %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
        },
        'filter':{   # optional
            '()': myFilter,    # use '()' to specify which class to use for filtering
            'name': 'param'    # value of name will be parse into Filter object when initialized
        }
        'handlers': { 
            'default': { 
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',  # Default is stderr
            },
            'file':{
                'level': 'WARNING',
                'formatter': 'standard',
                'class': 'logging.RotatingFileHandler',
                'filename': f'{BASE_DIR}/logs/myfile.log',
                    # BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
                'maxBytes': 1024*1000,  # 1 MB
                'when': 'D',   # slice logs every day, default 'D'   'W0' week
                'backupCount': 30    # keep logs 30 days(when param)
            },
            'file_handler': {
                'class': 'logging.FileHandler',
                'level': 'WARNING',
                'formatter': 'standard',
                'filename': os.path.join(LOGGING_DIR, 'file_io.log'),
                    # BASE_DIR = Path(__file__).resolve().parent.parent
                    # LOGGING_DIR = os.path.join(BASE_DIR, 'logs')
            },
        },
        'loggers': { 
            '': {  # root logger
                'handlers': ['default'],
                'level': 'WARNING',
                'propagate': False
            },
            'my.packg': { 
                'handlers': ['default'],
                'level': 'INFO',
                'propagate': False
            },
            '__main__': {  # if __name__ == '__main__'
                'handlers': ['default'],
                'level': 'DEBUG',
                'propagate': False
            },
        } 
    }
    
    class myFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if 'no_log_needed' in record.getMessage:
                return False   # don't log
            return True
    
    using:
    from logging.config import dictConfig
    logging.config.dictConfig(LOGGING_CONFIG) # Run once at startup:
    log = logging.getLogger(__name__)  # logger name input, '' using __name__,   logging.getLogger(my.packg)
    log.debug("Logging is configured.")
    
    
    # socket
    # server side
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 8001))
    sock.listen(5)
    while True:
        connection,address = sock.accept()   # connection is a socket object
        try:
            connection.settimeout(5)
            buf = connection.recv(1024)
            if buf == '1':
                connection.send(bytes('welcome to server!','utf-8'))
            else:
                connection.send(bytes('please go out!','utf-8'))
        except socket.timeout:
            print('time out')
        connection.close()
    # client side
    import socket  
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
    sock.connect(('localhost', 8001))  
    import time  
    time.sleep(2)  
    #sock.send(bytes('1','utf-8'))  
    print(sock.recv(1024))
    sock.close()  
    

    
    # setup mirrors in china
    cd .pip   ls  cat pip.conf        c:/Users/cai/pip/pip.ini  
    [global]
    index-url=https://pypi.doubanio.com/simple
    #index-url=https://mirrors.aliyun.com/pipy/simple/
    
    chinese character encode decode different rule show gibberish code 
    chinese character show ?? encoding wrong rule
    
    memory alignment: pointer read 4/8 bytes one time for 32/64 bit system, for multi-platform program, memory alignment
        happened when using datatype don't fill 4/8 byte followed by another datatype, to align memory, it move the next 
        data item to the start of next 4/8 place, instead of place next to the previous data. So that when read next 
        data, instead of reading 2 4/8 block and merge without alignment, it only need read once.
    
    
    """


if __name__ == '__main__':
    filename = ''
    s = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    for i in range(5):
        filename += s[random.randint(0, len(s) - 1)]
    print(filename)  # random character digit generator
    print(filename.count('1'))
    list1 = [1, '2', True]
    print(list1)

    md5 = hashlib.md5()
    with open('../resources/data/numpy_data.txt', mode='rb') as my_file:

        part = my_file.read(16)
        while part:
            md5.update(part)
            part = my_file.read(16)
    md5a = md5.hexdigest()
    print(md5a)

    with open('../resources/data/numpy_data.txt', mode='rb') as my_file:
        whole = my_file.read()
    md5b = hashlib.md5(whole).hexdigest()
    print(md5b)
    assert md5a == md5b


