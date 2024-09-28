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

#### Two Sum Problem*
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
#### Valid Palindrome*
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

#### Three Sum*
Given an array, find all unique triplets that sum upto zero.

##### Intuition
- Here we use can use the typical apprach of sorting and the two-pointer technique with a small modification.
- We fixate on one element and using the `two sum` approach we find the other two elements that sum upto zero.
- Also avoid duplicates by skipping over repeated elements, if the next is the same as the previous.

Code
```python
def three_sum(nums):
    nums.sort()
    result = []
    n = len(nums)

    for i in range(n):
        # Avoid duplicates
        if i > 0 and nums[i] == nums[i-1]:
            continue
        
        # Fix on element at i, go with two pointer on the rest of the array 
        left, right = i+1, n-1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append((nums[i], nums[left], nums[right]))
                left = left + 1
                right = right - 1

                # Avoid duplicates
                while left < right and nums[left] == nums[left-1]:
                    left = left + 1
                while left < right and nums[right] == nums[right+1]:
                    right = right - 1
            elif total < 0:
                left = left + 1
            else:
                right = right + 1
    return result
```

#### Container with Most Water*
Given an array of non-negative integers where each element represents the height of a vertical line on a graph, find two lines that together with the x-axis form a container that holds the most water.

##### Intuition
- Use **two pointers**, one at the beginning and one at the end of the array. Move the pointers towards each other to find the maximum area.
- The area is determined by the shorter line and the distance between the pointers.
- To move the pointers we move the one which has a smaller value - greedy approach to always ensure we have the larger one available to calculate the maximum window area.

Code
```python
def max_area(height):
    left, right = 0, len(height) - 1
    max_area = 0

    while left < right:
        # Calculate the area with the current left and right pointers
        width = right - left
        current_height = max(height[left], height[right])
        current_area = width * current_height

        # Update maximum area
        max_area = max(max_area, current_area)

        # Move the pointers based on the height comparison
        if height[left] < height[right]:
            left = left + 1
        else:
            right = right - 1
    
    return max_area
```

#### Trapping Rain Water*
Given an array of non-negative integers representing the height of bars in a histogram, find the total amount of water that can be trapped between the bars after raining.

##### Naive
###### Intution
- Iterate through each index and calculate the water trapped at that index as the minimum of the maximum heights to its left and right minus the height at that index.
- Sum up the trapped water at all indices.

```python
def trap(self, height: List[int]) -> int:  
    n = len(height)  
      
    # Arrays to store the maximum heights to the left and right of each index  
    max_left, max_right = [0] * n, [0] * n  
  
    # Fill the max_left array  
    for i in range(1, n):  
        max_left[i] = max(max_left[i - 1], height[i - 1])  
  
    # Fill the max_right array  
    for i in range(n - 2, -1, -1):  
        max_right[i] = max(max_right[i + 1], height[i + 1])  
  
    # Calculate the total trapped water  
    trapped_water = 0  
    for i in range(n):  
        # Water trapped at index i  
        water_at_i = max(min(max_left[i], max_right[i]) - height[i], 0)  
        trapped_water += water_at_i  
  
    return trapped_water
```

##### Two Pointer Approach
###### Intuition
- Two Pointers:
  - Use two pointers, left and right, starting at the beginning and end of the array, respectively.
  - Keep track of the maximum heights encountered so far from the left (left_max) and right (right_max).
- Calculate Trapped Water:
  - At each step, compare the heights at the left and right pointers.
  - Whichever side is lesser, move the pointer that side and calculate the trapped water by comparing with the maximum on that side.

Questions:
- **Why don't we consider the height on the other side for the trapped water?**
  We have already chosen the side which has the smaller height, thus removing the contention from the other side to be lesser. As the trapped rain water would depend upon the minimum from both sides.
  
- **Why do we move the pointer first and then calculate the trapped water?**
  This is because at the first and the last index the water trapped would always be 0 and we start our pointers from the end.

```code
def trap(height):
    n = len(height)
    left, right = 0, n - 1
    left_max, right_max = height[left], height[right]
    trapped_water = 0

    while left < right:
        if height[left] <= height[right]:
            left = left + 1
            left_max = max(left_max, height[left])
            trapped_water = trapped_water + max(left_max - height[left], 0)
        else:
            right = right - 1
            right_max = max(right_max, height[right])
            trapped_water = trapped_water + max(right_max - height[right], 0)
    
    return trapped_water
```

### Slow-Fast Pointer
Also known as the tortoise and hare technique, one pointer (the slow pointer) moves at a slower pace, while the other (the fast pointer) moves at a faster pace.
This is useful in solving several linked list problems.

#### Problems
#### Linked List Cycle Detection*

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
