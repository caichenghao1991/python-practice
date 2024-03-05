"""
    pip install jupyter
    jupyter notebook    # start notebook


    esc                 # exit to command mode
    ctrl+enter          # run cell
    #text               # title(bold) text
    help(len) / len?    # show document
    len??               # show source code
    tab                 # auto completion
    shift + double tab  # show function input parameter
    %run xxx.py         # run python script and import to notebook, can call function inside xxx.py directly
    %time  function()   # print execution time for one time
    %timeit function()  # run multiple time, take average run time
        timeit -r 5 -n 100 func()       # run timeit 5 times, function 100 times take average
    %who                # print environment variable, functions
        %who_ls         # put environment variable, functions names in list
    !dir                # use terminal command
    lsmagic             # show more magic command
    %alias??            # show all commands



    left click / enter  # enter coding mode
    dd                  # delete cell
    m                   # convert to markdown document
    *                   # code running
    a                   # add a cell above current
    b                   # add a cell below current
    up/down arrow       # move to above/below cell



"""
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.set_params()