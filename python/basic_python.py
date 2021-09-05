from decimal import *
import random


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
    mutable data types: list, dictionary, set and user-defined classes.
    data type can be changed once the value has been changed
    
    
    --Print and input
    print(object(s), sep=separator, end=end, file=file, flush=flush)
    print(a, b)  # default space separate variables, default new line use sep=',' to add separator  end='' no new line
    print(a,b,sep=',',end='')
    input(prompt string) allows user input. program pause until input received. always return string object
    
    
    --String
    quotes inside string need alternative double and single quotes or use \", \'  word = "He's cool"
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
    list2 = list1.copy()  or  list2 = list1[:] # deep copy, change in new list won't affect original 
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
    
    
    --Others 
    import random     random.randint(1, 10)  [1,10] random integer
    id(variable)  # get the readable memory location (integer) of the variable stored
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
    a = set(list1)
    print(a)

if __name__ == '__main__':
    review = Review()
