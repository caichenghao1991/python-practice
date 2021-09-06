from decimal import *
import random
from functools import reduce


class Review:
    """"""
    """
    --Naming convention
    variable name: letter number _, can't start with number, case sensitive
    better use student_name than studentName (camel case)
    class name: camel case, first character capitalized   class SchoolStudent:
    student_age, age = 3, 3   can declare multiple variable at same time
    a, b = b, a    swap a and b
    
    --Data type
    immutable data types: int, float, decimal, bool (True False), string, tuple, and range.
        immutable data type when value change, will create a new item in memory if that value doesn't already in memory,
        and assign new location to variable  
        so a=1;   b=a;   a=2;  print(b) # 1
    mutable data types: list, dictionary, set and user-defined classes.
        mutable data type when value change, will not affect the location it stored in
        li = [1,2];  li2=li;  li[0]=0;  li.append(3);  print(li2) # [0,2,3]
    data type can be changed once the value has been changed
    
    
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
    variable with same string value share the same memory location (reference), but change one won't affect other
    slicing: s[start:end:step]  [start,end) :step is optional.   start, end can be negative, last char has index -1. 
        start, end can be empty, default to 0 and last index
    functions:
    len   # len(var)  string length
    find    index    rfind   rindex     # var.find('_') return index of first _, return -1 if not found
                                        # var.find('_')  same as find, but will raise exception if not found
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
    capitalize title  upper lower upper    # var.title()  return every word first character capitalized  
                                        # var.capitalize()    return first character capitalized  
    ljust  rjust  center  lstrip  rstrip  strip  replace   # var.lstrip()  remove left spaces before 1st non-space char
                                        # var.center(30)   return 30 length string with original at center, rest spaces
                                        # var.rjust(30)  return 30 length string with original at right, rest spaces                                               
    join                                # ' '.join(iterable)   return string of contacting elements with space between
        
         
    --Operators
    arithmetic operator: +  -  *  /  **  //  %  
    assignment Operators: =  +=  -=  *=  /=  %=  //=  **=  &=  |=  ^=  >>=  <<=
    comparison operator: >  <  >=  <=  ==  !=  
    identity operator: is  is not
    membership operator: in  not in
    logical operator: and  or  not     print(1 and 3)  # 3
    bitwise operator:  &   |   ^ XOR(both true or both false return false)    ~ NOT   << (add 0 right most)    >>
    
    
    --Binary and nary 
    bin(149)  # 0b10010101  binary     oct(149)  #0o225  8     hex(149) # 0x95 16     int() # back to decimal
    -7 in binary: 7 in binary ob0000 0111,    reverse ^  ob1111 1000,  then +1  ob1111 1001     ~n+1 is -7 
    binary(negative) to decimal: ob1111 1001,   -1  ob1111 1000, then reverse ^ ob0000 0111  is 7
    print(~7+1)  # -7          n << 3    # 7 * 2^3               n >> 2   # 7 // 4
    
    
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
        pass   # run one time after for loop end
    for loop only for certain iterations, while loop can handle uncertain iterations
         
    
    --List
    mutable array able to store different data type entries
    Create / Copy
    list1 = []  list1 = [1, '2', True]    list2 = list(str)  # create char list from string
    list(range(3,5))  # [3,4]       [i * 2 for i in [1, 2, 3, 4] if i%2==0]  # [4, 8]      
    [i for i in 'hi']  # ['h', 'i' ]   
    [w.lower() if w.startswith('h') else w.upper() for w in list1]  list1 item start with h then lower otherwise upper 
    [(x,y) for x in range(2) for y in range(3)]  return all combination of (x,y)
    list2 = list1.copy()  or  list2 = list1[:] # deep copy, change in new list won't affect original, 
        # if list2 = list1, when list1 changed, it will change list2 as well since they have same memory location
    Read
    print(list1[0], list1[:1], list1[::-2])  # [True, 1]     
    len(list1) #3    
    list1.index(1)  # 0  exception if not found    list1.index(1,0,2)  find 1 in index between [ 0,2)
    list1.count(1)  # 1  number of appearance of value
    Add   mutate list
    list1 * 2  # [1, '2', True, 1, '2', True]
    list1 + [100]    # [1, '2', True, 100]  # not mutate original list
    list1.append(100)  # [1, '2', True, 100]   
    list1.extend([100, 200])  # [1, '2', True, 100, 200]   # iterable input parameter
    list1.insert(1, 300)   # [1, 300, '2', True]  # insert item at index, shift one to the right after index 
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
    
    
    --Dictionary
    Also known as map or hashtable (key value pair), retain order of insertion when iterating keys after Python 3.7
    dictionary key need to be immutable: int, float, boolean, string, None, tuple
    same dictionary key, the value will be overwrote    use hash O(1) operations
    Create
    my_dict = {'name': 'Andrei Neagoie', 'age': 30, 18: False}    dict1 = {}   dict2 = dict1.copy()
    dict2 = dict([('name','Harry'),('age',[10,1])])
    dict2 = dict.fromkeys(['name','age'], 1)   # create dictionary with keys and same default value   
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
    name = my_dict.pop('name', None)  # delete key name, return value of that key. keyError if key not exist
    (k, v) = my_dict.popitem()   # delete last item, if no item exist throw keyError 
    dict1.clear()  # clear whole dictionary, empty dictionary remained
    del dict1  # delete the dictionary structure as well
    dict can convert to list tuple set but only keep keys           
    
    --Set
    Unordered collection of unique item, no order, use hash   use hash O(1) operations
    Create
    set1 = set()  set1, set2 = {1,2,3},{3,4,5}    set(new_list)    new_set = set1.copy()  
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
        return kwargs['name'], return kwargs['date']    can have multiple return statement under various if condition
    name = function_name3(1, 2,**{'name':'JK','date':'1991-10'})  # name is 'JK' for first return  
    item = function_name3(1, 2,**{'name':'JK','date':'1991-10'})  # item: ('JK','1991-10')  put return items in a tuple
    name,date = function_name3(1, 2,**{'name':'JK','date':'1991-10'})  # name is 'JK', date is '1991-10' for 2nd return
    
    if immutable variable var, inside function, add: global var    var = 7  to change the global outside variable values
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
            nonlocal a    # a cannot be modified before nonlocal declaration
            a = 200  # inner function can't modify outer function variable, it only create a new inner variable when
            print(a+10)     # have the same name as outer variable, unless add nonlocal var declaration   
                         # when variable is used, find declaration with order: inner -> outer -> global -> builtin    
        print(inner)  #return location info
        info = locals()  # info will have local variable and value, inner fuction location information
        inner()  # run inner function
        or return inner
    r = outer()  # return inner function memory location
    r()  # run inner function   or  outer()()
    
    info = globals()  return the global variable (system and user-defined) dictionary 
    
    decorator pattern
    def d(func):      
        print('<-')
        def wr(para):      # def wr(*args,**kwargs):  to cover all input cases
            func(para)     # func(*args,**kwargs) 
            print(1)
            return 'x'
        print('->')
        return wr   #
    @d      # <- ->  equivalent to:  f = d(f), used for add additional logic while keep original function name and call
    def f(para):
        print(2)  
        return 'x'
    r = f(para_val)   # 2  1  # first execute outer function of d then go into inner function
                  # equivalent to call wrapper function
                  
    decorator function can have input parameters as well need extra layer of outer function
    def outer_param(para):
        def d(func):  # same content
        return d
    @outer_param(para_val)
    def f(para):
    
    
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
    list1 = [('sam',35),('tom',19), ('tony',20)] 
    print(max(list1, key=lambda x: x[1]))   # ('sam',35)   key is a function x is each tuple in list1                              
    max, min, sorted all have key function
    print(list(filter(lambda x:x[1]>19, list1)))  # [('sam', 35), ('tony', 20)] filter out data not match condition
    print(list(map(lambda x:x[0].title(), list1)))  # ['Sam', 'Tom', 'Tony']  retrieve part of data
    from functools import reduce
    print(reduce(lambda x, y: x+y, [1,2,3,4,5]))  # sum of array, or concat strings
    r = zip([1,2,3], {'one','two','three'});  print(set(r))  # {(1, 'one'), (2, 'two'), (3, 'three')}
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
    'w+' - Read and write (overwrite from beggining).
    'r+' - Read and write from the start.
    'a+' - Read and write from the end.
    't' - Text mode (default).
    'b' - Binary mode.  (audio, video, image...)
    
    my_file = open('test.txt')    open('path/test.txt', mode = 'a', buffering, encoding)  return a steam object
        # FileNotFoundError if path not exist
    print(my_file.readable())    # return boolean whether stream is readable
    print(my_file.read()) #read whole file
    my_file.seek(0)  # reposition cursor to start so can read again
    my_file.readline()  #read one line and move cursor down 
    my_file.readlines()  # return list of lines
    my_file.write(string)   # in 'w' mode will clear content then write
    my_file.writelines(Iterable)   # need add  \n in string to switch new line
    my_file.close()  # release resource
    or use with open() as stream to auto release resource   
    try:
        with open('path/test.txt', mode = 'a') as my_file:    
            # windows       open(r'C:\path\test.txt', mode = 'a')   open('C:\\path\\test.txt')
            #  ./ from current folder      #  ../ back one folder
            my_file.write("Hello World")
            print(my_file.readlines())
        except FileNotFoundError, IOError as err:
            print('file does not exist')

    import os   
    os.path.isabs(string)   # return bool check path is absolute path
    os.path.abspath('basic_python.py')  # return file name's absolute path
    os.path.abspath(__file__)  # return file name's absolute path
    os.path.dirname(__file__)  # return string of current file's absolute directory
        os.getcwd()  # return string of current file's absolute directory
    os.path.split(path)   # return tuple of directory and file name  ('C:\\path\\','a.txt')
    os.path.splitext(path)  # return tuple of file extension and directory+file name  ('C:\\path\\a','.txt')
    os.path.getsize(path)   # return file size in bytes
    os.path.join(os.getcwd(),'path','a.txt')     # return path with path\\a.txt inside current directory
    os.path.listdir(path)   # return a list containing all the file and directory under path
    os.path.exists(path)   # return bool   check path exist or not
    os.path.isfile(path)   # check path is file or not
    os.path.isdir(path)   # check path is directory or not
    os.mkdir(path)   # no return  raise FileExistError if path already exist
    os.rmdir(path)   #   remove empty directory   otherwise raise OSError
    os.remove(path)   # remove file 
    os.chdir(path)   # switch current working directory to path
    
    
    --Exceptions
    
    
    --Others 
    import random     random.randint(1, 10)  [1,10] random integer
    id(variable)  # get the readable memory location (integer) of the variable stored
    isinstance(var, int)   # return bool   check whether variable data type is integer
    import sys    sys.getrefcount(var)   # return the number of variable using the reference var
    
    
    """

    student_age, age = 3, 3

    print(type(student_age))  # <class 'int'>
    student_age = 3.5
    print(type(student_age))  # <class 'float'>
    pi = Decimal('1.1')  # used to avoid inaccuracy   float might get 1.1000000000000001

    print(type(pi))
    word = "He's a \"genius\"."
    poem = '''
                Night Thought
                    Li Bai           
    '''
    print(word)
    print(poem)
    # age = input("Please Enter your age:")   # always return string object
    print("You are %d years old." % int(student_age))  # % (age, name)
    print(student_age, student_age)  # default space separate, default new line
    # print(student_age, age, sep=',', end='') # use sep=',' to add separator  end='' no new line
    print(0 <= int(student_age) <= 100)
    print(int(0x95))
    n = 7
    print(~n + 1)  # -7
    print(n << 3)  # 7 * 2^3
    print(n >> 2)  # 7 // 4
    print(4 if age > 5 else age)

    for i in range(1, 10, 3):
        print(i)
    else:
        print("end")
    filename = ''
    s = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    for i in range(5):
        filename += s[random.randint(0, len(s) - 1)]
    print(filename)  # random character digit generator
    print(filename.count('1'))
    list1 = [1, '2', True]
    print(list1)


if __name__ == '__main__':
    aaa = 1
    review = Review()
    list1 = [1,2,3,4]

    print(reduce(lambda x, y: x+y, ['1','2','3']))

