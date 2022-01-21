'''
array:
    good:   1 (hashmap, no need 2 pass)
            11 (2 pointer both side, eliminate lower item because all combination inward has lower volume than
                current, since width smaller, and height either same or lower)
            26 only need find result between [1, len(arr)+1]
            31
    review:
            15 use map (key: value, value: count of occurrence of that value), double loop in map, check desired value
                key exist
    bad:

2 pointer/sliding window
    good:   42 2 pointer left right, record left_max and right_max, meet at middle, move smaller max pointer, add res
                with min(left_max, right_max)-height[curr] if height[curr]< height_max
    review: 15 another approach: 1st pointer iterate through item, 2nd and 3rd pointer from left and right edge on the
                right side of 1st pointer, moving towards each other. need remove duplicate(1st pointer skip same value,
                when matched 2nd continue move right if same next value, 3 rd pointer move left if same next value, then
                must move 2nd pointer right 1 step avoid infinite loop)

    bad:


stack
    good:
    review:  42 another approach: height append 0, stack initialize -1. during continuous pop if current item larger
                than peak item, need check stack size since need remaining peak value after pop. compare to problem 84,
                no need check empty stack because 84 pop if current item smaller than peak item
    bad:    84 monotonous stack, between current index and the popped item index are all greater than the popped item
                value. so area will be (current index - stack peak (item) index - 1)* popped item height
                ex: height: [4, 5,5,3]   when met 3(index 3), pop 5, stack left 4's index, so width 3-0-1=2
                add a height 0 at end and init stack -1 (compare to the last 0 height item)

binary search:
    good:   33: first half if condition search the certain monotonous increase section, rest leave in else part
    review:
    bad:


back track
    good:
    review:
    bad:

dp
    good:
    review:
    bad:
'''
l=[1,11]
print(sorted(l))