# Two Pointers
The idea is to use two pointers to traverse the data structure in a coordinated way, often to achieve a linear time complexity solution.

## Variations
### Generic Two-Pointers
In generic two-pointers, we have two pointers traversing the array and either moving towards or away from each other to satisfy a particular condition. The pointers represent a `pair` of elements.

#### Identification
This technique is particularly useful for problems involving pairs, such as finding pairs that sum to a particular value or checking if a string is a palindrome.

#### Flow
- Initialize the pointers
- Move the pointers
- Identify the stopping condition

#### Two Sum Problem
Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number. The function should return indices of the two numbers such that they add up to the target.

By default the solution that we would think is to iterate through all possible pairs and return the indices of the ones that add upto the target.

```python
def two_sum(arr, target):
    n = len(arr)
    for i in range(n):
        for j in range(n):
            if arr[i] + arr[j] == target:
                return (i, j)
    return (-1, -1)
```

This solution though would have a time complexity of `O(n^2)`. Can we make it better?

##### Intuition
- **Sorted Array Advantage**: Utilize the sorted order of the array to guide the search for the target sum.
- **Pointer Movement**: Initialize two pointers—left at the start and right at the end. Compute the sum of elements at these pointers.
- **Adjust Pointers**: 
  - If the sum is equal to the target, return the indices. 
  - If the sum is less than the target, increment the left pointer to increase the sum. 
  - If the sum is greater, decrement the right pointer to decrease the sum. Continue until the pointers converge.

Update code
```python
def two_sum(arr, target):
    n = len(arr)
    left, right = 0, n

    while left < right:
        s = arr[left] + arr[right]
        if s == target:
            return (left, right)
        elif s > target:
            right = right - 1
        else:
            left = left + 1
    
    return (-1, -1)
```

#### Problems
#### Valid Palindrome
Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.

##### Intuition
- We start comparing from both ends to check whether the characters are same, break if it doesn't match till we converge the pointers.
- Here converge means that the pointers have passed each other as even the pointers pointing to the same element is a valid condition for odd-length palindromes.
- There is also an approach where you start from the middle and go to the both ends but little lesss intuitive.

```python
def valid_palindrome(string):
    n = len(string)
    left, right = 0, n-1

    while left <= right:
        if string[left] != string[right]:
            return False
        left = left + 1
        right = right - 1
```

#### Three Sum
```
// TODO
```

### Slow-Fast Pointer
Also known as the tortoise and hare technique, one pointer (the slow pointer) moves at a slower pace, while the other (the fast pointer) moves at a faster pace.

This is useful in solving several linked list problems.

#### Problems
#### Linked List Cycle Detection

##### Intuition
- Different Speeds: Use two pointers, slow and fast. The slow pointer moves one step at a time, while the fast pointer moves two steps at a time.
- Cycle Detection:
  - If there is no cycle in the linked list, the fast pointer will eventually reach the end (null) without ever meeting the slow pointer.
  - If there is a cycle, the fast pointer will eventually "lap" the slow pointer, meaning they will meet at some point within the cycle.

Code
```python
def has_cycle(head)
    slow, fast = head, head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            return True
    
    return False
```

Similar problem.
#### Middle of the Linked List
Given a non-empty, singly linked list with head node head, return a middle node of the linked list. If there are two middle nodes, return the second middle node.

##### Intuition
- Different Speeds: Use two pointers, slow and fast. The slow pointer moves one step at a time, while the fast pointer moves two steps at a time.
- When fast reaches the end, the slow would be at the **middle** of the linked list.

Code
```python
def middle_node(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```
