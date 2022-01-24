'''
array:
    good:   1 (hashmap, no need 2 pass)
            26 only need find result between [1, len(arr)+1]
            31
            75  count 0,1 then reorder
            88
            287: change item with index of current value to negative, when check value with index of current value, if
                it's negative number, then return this duplicate value. need use nums[abs(val)] since value can be
                negative and only check nums[abs(val)]<0

    review:
            15 use map (key: value, value: count of occurrence of that value), double loop in map, check desired value
                key exist
    bad:

2 pointer/sliding window
    good:   11 (2 pointer both side, eliminate lower item because all combination inward has lower volume than
                current, since width smaller, and height either same or lower)
            42 2 pointer left right, record left_max and right_max, meet at middle, move smaller max pointer, add res
                with min(left_max, right_max)-height[curr] if height[curr]< height_max
            76  care with indent level
    review: 3  check v inside dic and pre pointer <= dic[v] update pre pointer to dic[v]+1.
            15 another approach: 1st pointer iterate through item, 2nd and 3rd pointer from left and right edge on the
                right side of 1st pointer, moving towards each other. need remove duplicate(1st pointer skip same value,
                when matched 2nd continue move right if same next value, 3 rd pointer move left if same next value, then
                must move 2nd pointer right 1 step avoid infinite loop)
            30  another approach: need pay attention to duplicate words inside words list, sliding window faster than
                linear scan
    bad:

string:
    good:   5  create a function check(str,i,j) so can deal with i=j or j=i+1
            6  be careful edge case numRows=1
            30 scan each index verify dic[word:count] all values 0 at end of loop
    review: 3 if character not seen, add 1 for temp_max, add to set, otherwise, update res with temp_max, keep discard
                pointer(init at 0) item and move pointer right one step, and temp_max -1 until met the same character as
                current character, and move pointer right one more step. return max(res, temp_max) (longest substring
                end at last character)
    bad:

linked list:
    good:   2  pay attention to last remaining 1 after both list finished.  dummy = ListNode(None); curr=dummy
                while l1 or l2:  curr.next = ListNode(v); curr=curr.next    # for adding node
            19 dummy = ListNode(None); dummy.next=head;  curr=dummy;  while curr    # for traverse
            24 dummy = ListNode(0, head)  curr = dummy.next;  pre =dummy; while curr and curr.next:  # swap node need
                previous node info
            23 heap
            92
    review:
    bad:


stack
    good:   32 initialize temp list 0, use stack push index instead of character, when ')' pop item. change temp curr
                index and popped index to 1. finally count consecutive 1 in temp list
    review: 42 another approach: height append 0, stack initialize -1. during continuous pop if current item larger
                than peak item, need check stack size since need remaining peak value after pop. compare to problem 84,
                no need check empty stack because 84 pop if current item smaller than peak item
    bad:    84 monotonous stack, between current index and the popped item index are all greater than the popped item
                value. so area will be (current index - stack peak (item) index - 1)* popped item height
                ex: height: [4, 5,5,3]   when met 3(index 3), pop 5, stack left 4's index, so width 3-0-1=2
                add a height 0 at end and init stack -1 (compare to the last 0 height item)

binary search:
    good:   33 first half if condition search the certain monotonous increase section, rest leave in else part

    review:
    bad:

tree:
    good:   100 condition similar as 101
            103
    review: 95 parameter: start, end, return all_trees (must return since need results from left and right child)
                base case: if start>end: return [None]
            101 if both child empty return true, if left, right value same then return left and right recursive result
                else return False
    bad:

back track
    good:
    review: 78  no need parameter length(result can have any length), since unique items,just append till end of list
            90  check duplicate exclude item with increase length with duplicate of previous digit (ex. 122) by only
                continue when start index in recursive parameter > index in for loop
    bad:

graph:
    good:   79 need change visited node value to '#' otherwise will revisit node in the path already. or keep visited
                matrix
    review:
    bad:
dp
    to be done:   32
    good:
    review: 5 initialize 1 and 2 character case,then dp[i][j]=True means s[i:j+1] is valid  dp[i][j]=dp[i+1][j-1]
                don't forget update max length
    bad:
'''
l=[1,11]
print(sorted(l))