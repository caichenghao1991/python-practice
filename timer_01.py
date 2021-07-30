from time import time

names = ['Dory','Bruce','Nemo','Marlin']
test_list = ['Nemo']*100000

def findNemo(list):
    '''
        find string nemo in a list of strings
        O(n) time complexity
    '''
    start_time = time()
    for _ in list:
        if _ == 'Nemo':
            print('Found Nemo')
    duration = time() - start_time
    print('Method 1 call to find Nemo took ' + str(duration) + ' milliseconds')


if __name__ == '__main__': # Runs main() if file wasn't imported.
    # findNemo(names)
    findNemo(test_list)
