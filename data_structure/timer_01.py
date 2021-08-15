from time import time

names = ['Dory', 'Bruce', 'Nemo', 'Marlin']
test_list = ['Nemo'] * 100000


def find_nemo(li):
    """
        find string nemo in a list of strings
        O(n) time complexity
        O(1) space complexity, does not consider input memory cost
    """
    start_time = time()
    for _ in li:
        if _ == 'Nemo':
            print('Found Nemo')
    duration = time() - start_time
    print('Method 1 call to find Nemo took ' + str(duration) + ' milliseconds')


def find_nemo2(list1, list2):
    """
        find string nemo in 2 lists of strings
        O(a+b) time complexity
    """
    start_time = time()
    for _ in list1:
        if _ == 'Nemo':
            print('Found Nemo')
    for _ in list2:
        if _ == 'Nemo':
            print('Found Nemo')
    duration = time() - start_time
    print('Method 2 call to find Nemo took ' + str(duration) + ' milliseconds')


if __name__ == '__main__':  # Runs main() if file wasn't imported.
    # find_nemo(names)
    find_nemo(test_list)
    find_nemo2(test_list, test_list)
