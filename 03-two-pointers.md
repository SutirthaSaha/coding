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

#### [Two Sum Problem](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted)*
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
#### [Three Sum](https://leetcode.com/problems/3sum)*
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

#### [4 Sum](https://leetcode.com/problems/4sum)*
Given an array `nums` of `n` integers, return an array of all the unique quadruplets `[nums[a], nums[b], nums[c], nums[d]]` such that:
- `0 <= a, b, c, d < n`
- `a`, `b`, `c`, and `d` are distinct.
- `nums[a] + nums[b] + nums[c] + nums[d] == target`
You may return the answer in any order.

##### Intuition
- **Sorted Array Advantage**: Start by sorting the array. Sorting helps to systematically reduce the problem size and avoid duplicates.
- **Recursive Decomposition**: Use a recursive function to reduce the k-sum problem to a simpler problem. Specifically, break down the 4-sum problem into smaller subproblems until reaching the 2-sum problem, which can be efficiently solved using the two-pointer technique.
- **Base Case - Two Sum**:
  - Use two pointers to find pairs in the sorted array that sum up to the target.
  - Initialize two pointers: `left` at the start and `right` at the end of the array segment.
  - Compute the sum of elements at these pointers.
    - If the sum is equal to the target, add the pair to the result.
    - If the sum is less than the target, increment the left pointer to increase the sum.
    - If the sum is greater than the target, decrement the right pointer to decrease the sum.
  - Continue until the pointers converge.
- **Recursive Step**:
  - For k > 2, iterate through the array, fix one element, and recursively solve the (k-1)-sum problem for the remaining elements.
  - Ensure to skip duplicates to avoid repeating quadruplets.
- **Combining Results**: The recursive function combines results from the base case to form valid quadruplets.

Code
```python
def fourSum(nums, target):
    result = []
    nums.sort()
    n = len(nums)
    def solve(left, target, k, curr):
        if k == 2:
            right = n-1
            while left < right:
                total = nums[left] + nums[right]
                if total == target:
                    result.append(curr + [nums[left]] + [nums[right]])
                    left = left + 1
                    right = right - 1

                    while left < right and nums[left] == nums[left-1]:
                        left = left + 1
                    while left < right and nums[right] == nums[right+1]:
                        right = right - 1
                elif total < target:
                    left = left + 1
                else:
                    right = right - 1
        else:
            for index in range(left, n-k+1):
                if index > left and nums[index] == nums[index - 1]:
                    continue
                target = target - nums[index]
                curr.append(nums[index])
                solve(index+1, target, k-1, curr)
                target = target + nums[index]
                curr.pop()
    
    solve(0, target, 4, [])
    return result
```

**This also gives you generic solution for *n Sum***

#### [Valid Palindrome](https://leetcode.com/problems/valid-palindrome)*
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

#### [Container with Most Water](https://leetcode.com/problems/container-with-most-water)*
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

#### [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water)*
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
  - Whichever side is lesser, calculate the trapped water by comparing with the maximum on that side and move the pointer that side.

Questions:
- **Why don't we consider the height on the other side for the trapped water?**
  We have already chosen the side which has the smaller height, thus removing the contention from the other side to be lesser. As the trapped rain water would depend upon the minimum from both sides.

Code
```python
def trap(height):
    n = len(height)
    left, right = 0, n-1
    left_max, right_max = height[left], height[right]
    trapped_water = 0

    while left <= right:
        if height[left] <= height[right]:
            left_max = max(left_max, height[left])
            trapped_water = trapped_water + left_max - height[left]
            left = left + 1
        else:
            right_max = max(right_max, height[right])
            trapped_water = trapped_water + right_max - height[right]
            right = right - 1
    
    return trapped_water
```

#### The Celebrity Problem
In a party of n people, a celebrity is defined as someone who is known by everyone but knows no one. You are given a matrix `M` of size `n x n` where `M[i][j]` is 1 if person `i` knows person `j`, otherwise it is 0. Implement a function `findCelebrity` that determines if there is a celebrity in the party. If there is a celebrity, return their index (0-based index). If there is no celebrity, return -1.

Example:
```
Input:  
M = [[0, 1, 0],  
     [0, 0, 0],  
     [0, 1, 0]]  
Output: 1  
```
Explanation:  
Person 1 is known by everyone but does not know anyone.

##### Naive Approach
We maintain 2 arrays - `person_know` and `others_know` array and keep populating as we traverse the entire matrix.
After traversing we now traverse these 2 arrays index by index and the index where `person_know` has `0` and `others_know` has `n-1` - the person at that index is a celebrity.
**There can never be 2 celebrities.**

Code
```python
def celebrity(matrix):
    n = len(matrix)
    person_know = [0] * n
    others_know = [0] * n

    for row in range(n):
        for col in range(n):
            if matrix[row][col] == 1:
                person_know[row] = person_know[row] + 1
                others_know[col] = others_know[col] + 1
    
    for i in range(n):
        if person_know[i] == 0 and others_know[i] == n-1:
            return i
    
    return -1
```

This approach however has a complexity of `O(n*n)`, we can utilise the 2-pointer approach to solve this in `O(n)`.

##### Two-pointer Approach
Given the properties, we can use a two-pointer approach to efficiently narrow down the potential celebrity:
- Initialization: Start with two pointers, left and right, representing the range of people we are considering as potential celebrities.
- Elimination Process:
  - Compare the people at the left and right pointers.
  - If left knows right, then left cannot be the celebrity, so we eliminate left and move the left pointer one step to the right.
  - If left does not know right, then right cannot be the celebrity, so we eliminate right and move the right pointer one step to the left.
- Convergence:
  - Continue this process until the left and right pointers converge to a single person.
  - At this point, the remaining person is our potential celebrity candidate.

Code
```python
def celebrity(matrix):
    n = len(matrix)
    left, right = 0, n-1

    while left < right:
        if matrix[left][right] == 1:
            # left knows right, so left cannot be a celebrity
            left = left + 1
        else:
            # left does not know right, so right cannot be a celebrity
            right = right - 1
    
    candidate = left

    for i in range(n):
        if i != candidate:
            # Candidate should not know anyone else
            if matrix[candidate][i] == 1:
                return -1
            # Candidate should be known by everyone else
            if matrix[i][candidate] == 0:
                return -1
    
    return candidate
```

### Slow-Fast Pointer
Also known as the tortoise and hare technique, one pointer (the slow pointer) moves at a slower pace, while the other (the fast pointer) moves at a faster pace.
This is useful in solving several linked list problems.

#### Problems
#### [Linked List Cycle Detection](https://leetcode.com/problems/linked-list-cycle)*
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

#### [Happy Number](https://leetcode.com/problems/happy-number)*
Write an algorithm to determine if a number `n` is happy.

A **happy number** is a number defined by the following process:
- Starting with any positive integer, replace the number by the sum of the squares of its digits.
- Repeat the process until the number equals 1 (where it will stay), or it **loops endlessly in a cycle which does not include 1**.
- Those numbers for which this process **ends in 1** are happy.

Return `true` if `n` is a *happy number*, and `false` if not.

##### Intuition
The problem of determining whether a number is a happy number can also be effectively solved using the slow-fast pointer approach, similar to detecting cycles in a linked list. This approach uses two pointers moving at different speeds to detect cycles.

**Key Idea:**
- Use two pointers, slow and fast.
- slow moves one step at a time (calculates the sum of squares once).
- fast moves two steps at a time (calculates the sum of squares twice).
- If slow or fast reaches 1, then the number is a happy number.
- If there is a cycle, slow and fast will eventually meet - not happy number.

Code
```
def isHappy(n):
    def get_digit_square_sum(n):
        total = 0
        while n:
            total = total + (n % 10)**2
            n = n // 10
        return total
    
    slow, fast = get_digit_square_sum(n), get_digit_square_sum(get_digit_square_sum(n))

    while fast != 1 and slow != fast:
        slow = get_digit_square_sum(slow)
        fast = get_digit_square_sum(get_digit_square_sum(fast))
    
    return fast == 1
```

#### [Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number)
Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.
There is only one repeated number in nums, return this repeated number.
You must solve the problem without modifying the array nums and using only constant extra space.

Example:
```
Input: nums = [1,3,4,2,2]
Output: 2
```

##### Intuition
The problem of finding the duplicate number in an array can be solved using the slow-fast pointer approach because the values are within the range `[1, n]`, each value can be used as an index to access the array. This allows us to traverse the array in a manner similar to traversing a linked list. The presence of a cycle in this array indicates a duplicate number.

Key Idea:
- Use two pointers, slow and fast.
- slow moves one step at a time.
- fast moves two steps at a time.
- Since there is a duplicate number a cycle exists and the slow and fast will eventually meet.
- Once they meet, reset one pointer to the start and move both pointers one step at a time to find the entrance to the cycle, which is the duplicate number.

Code
```python
def findDuplicate(nums):
    slow, fast = nums[0], nums[0]

    while True:
        slow, fast = nums[slow], nums[nums[fast]]
        if slow == fast:
            break
    
    pointer = nums[0]

    while slow != pointer:
        slow, pointer = nums[slow], nums[pointer]
    
    return pointer
```
