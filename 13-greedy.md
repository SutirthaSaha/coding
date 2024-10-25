# Greedy
Greedy algorithms are a class of algorithms that build up a solution piece by piece, always choosing the next piece that offers the most immediate benefit. The **hope** is that by making **locally optimal choices** at each step, the **overall solution will be optimal as well**.

## Characteristics
- **Greedy Choice Property**: The algorithm makes a series of decisions by choosing the best available option at each step without considering future consequences. This choice must be locally optimal and irrevocable.
- **Optimal Substructure**: The problem can be broken down into smaller subproblems, and the optimal solution to the problem can be constructed from the optimal solutions to its subproblems.

## Limitations
- **Non-Optimal Solutions**: Greedy algorithms do not always produce the optimal solution. They work well for problems with the greedy choice property, but may fail for others.
- **Local Optima**: Greedy algorithms can get stuck in local optima, missing the global optimal solution.

## Problems
### Activity Selection Problem
Given a set of activities with start and end times, select the maximum number of activities that don't overlap.

#### Strategy
Always select the activity that finishes first. This way, you leave as much room as possible for the remaining activities.

Code
```python
def activity_selection(activities):
    # Sort activities by end time
    activities.sort(key=lambda activity: activity[1])

    selected_activities = [activities[0]]

    for activity in activities[1:]:
        # If the start time of the current activity is greater than or equal to the end time of the last selected activity
        if activity[0] >= selected_activities[-1][1]:
            # Select the current activity
            selected_activities.append(activity)
    
    return selected_activities
```

This problem is similar to the `Non Overlapping Intervals` problem in the intervals section.

### Fractional Knapsack Problem
Given a set of items, each with a weight and a value, determine the maximum value you can carry in a knapsack with a weight limit, where you can take fractions of items.

#### Strategy
Calculate the value-to-weight ratio for each item and sort them in descending order. Then, take as much of the highest ratio item as possible, followed by the next highest, and so on.

Code
```python
def fractional_knapsack(weights, values, capacity):
    n = len(weights)
    
    # Calculate value to weight ratio for each item  
    ratios = [(values[i] / weights[i], weights[i], values[i]) for i in range(n)]

    # Sort items by value to weight ratio in descending order
    ratios.sort(reverse=True, key = lambda ratio: ratio[0])

    total = 0.0
    for ratio, weight, value in ratios:
        if capacity >= weight:
            # Take the whole item
            capacity = capacity - weight
            total_value = total_value + value
        else:
            # Take the fraction of item that fits
            total_value = total_value + ratio * capacity
            break
    
    return total
```

### Job Sequencing Problem
Given a set of jobs where each job has a deadline and a profit, schedule jobs to maximize total profit. Each job takes one unit of time and a job can only be scheduled if it can be completed by its deadline.

#### Strategy
Sort the jobs in descending order of profit. Then, try to schedule each job in the latest possible time slot before its deadline.

Code
```python
def job_sequencing(deadlines, profits, m):
    n = len(deadlines)
    jobs = [(i, deadlines[i], profits[i]) for i in range(n)]

    # Sort jobs by profit in descending order
    jobs.sort(key=lambda job: job[2], reverse=True)

    # Initialize a result array and a slot array to keep track of free time slots
    result = [None] * m
    slots= [False] * m

    # Iterate over sorted jobs
    for job in jobs:
        job_id, deadline, profit = job

        # Find a free slot for this job (starting from the latest possible slot)
        for free_slot in range(min(m, deadline), -1, -1):
            if slots[free_slot] is False:
                slots[free_slot] = True
                result[free_slot] = job_id
                break
    
    # return the jobs that could be scheduled
    return [job_id from job_id in result if job_id is not None]
```

### Dijkstra's Shortest Path Algorithm
In graph section refer `Single Source Shortest Path`.

### Prim's Minimum Spanning Tree Algorithm
In graph section refer `Minimum Spanning Tree`.

### Kadane's Algorithm
Kadane's algorithm is a famous algorithm used to solve the `Maximum Subarray Problem` in linear time.

#### Problem
The problem is to find the contiguous subarray within a one-dimensional array of numbers - **both positive and negative numbers** - that has the largest sum.

#### Intuition
Kadane's Algorithm makes a greedy choice at each step by selecting the option that maximizes the sum up to the current element. This choice ensures that the local maximum is always the best option for achieving the global maximum.

##### Working:
- The algorithm keeps track of two key pieces of information: the maximum sum of the subarray that ends at the current position and the maximum sum of any subarray encountered so far.
- As you iterate through the array, for each element, you decide whether to include it in the current subarray or to start a new subarray with this element. This decision is based on which option yields a higher sum.
  - If including the element in the current subarray results in a higher sum, you continue with the current subarray.
  - If starting a new subarray with the current element results in a higher sum, you start a new subarray.
- Compare this value with the global maximum and update if needed.

Code
```python
def maximum_subarray_problem(nums):
    max_ending_here = nums[0]
    max_subarray = nums[0]

    for num in nums[1:]:
        max_ending_here = max(num, max_ending_here + num)
        max_subarray = max(max_subarray, max_ending_here)
    
    return max_subarray
```

#### Problems based on Kadane's Algorithm
- Maximum Product Subarray

#### Maximum Product Subarray
Given an array of integers, find the contiguous subarray within the array that has the largest product.

##### Intuition
The intuition behind the Maximum Product Subarray is similar to Kadane's Algorithm, but with a twist as the product of two negative numbers is positive, you need to keep track of both the maximum and minimum products upto the current position.

##### Working
- The algorithm keeps track of three key pieces of information:
  - The maximum product of the subarray that ends in the current position
  - The minimum product of the subarray that ends in the current position
  - The maximum product subarray encountered so far.
- As you iterate through the array, for each element: 
  - Swap the current max and current min if the element is negative
  - Update them by considering the current element and their product with the current element.
  - Update the global maximum if needed.

Code
```python
def maximum_product_subarray(nums):
    max_product = min_product = result = nums[0]

    for num in nums[1:]:
        if num < 0:
            # TWIST: negative multiplied by negative is positive
            max_product, min_product = min_product, max_product
        
        # same old same old
        max_product = max(num, max_product * num)
        min_product = max(num, min_product * num)

        result = max(result, max_product)
```

#### Hand of Straights
Alice has some number of cards and she wants to rearrange the cards into groups so that each group is of size groupSize, and consists of groupSize consecutive cards.
Given an integer array `hand` where `hand[i]` is the value written on the `ith` card and an integer `groupSize`, return `true` if she can rearrange the cards, or `false` otherwise.

Example:
```
Input: hand = [1,2,3,6,2,3,4,7,8], groupSize = 3
Output: true
Explanation: Alice's hand can be rearranged as [1,2,3],[2,3,4],[6,7,8]
```

##### Intuition
The problem is essentially about checking if the cards can be divided into groups of consecutive numbers of a specified size. 
- To achieve this, we can use a greedy approach combined with:
  - a counting mechanism to track the frequency of each card
  - a min-heap to ensure we process the smallest card values first.
- Every time a card count reaches 0, it should be the top of the min heap - otherwise grouping wouldn't be possible.

Code
```python
def is_n_straight_hand(hand, group_size):
    # If the total number of cards is not divisible by groupSize, we cannot form the required groups
    if len(hand) % group_size:
        return False
    count = defaultdict(int)
    for card in hand:
        count[card] = count[card] + 1
    min_heap = list(count.keys())
    heapq.heapify(min_heap)

    # Process the cards starting from the smallest value
    while min_heap:
        # Get the smallest card value
        first = min_heap[0]

        # Try to form a group starting from the smallest card
        for card in range(first, first+group_size):
            if card not in count:
                return False
            count[card] = count[card] - 1
            # If the count of the `card` becomes zero, remove it from the heap and count dictionary
            if count[card] == 0:
                if card != min_heap[0]:
                    return False
                count.pop(card)
                heapq.heappop(min_heap)
    
    return True
```


#### Merge Triplet to Form Target Triplet
A triplet is an array of three integers. You are given a 2D integer array triplets, where `triplets[i] = [ai, bi, ci]` describes the `ith` triplet. You are also given an integer array `target = [x, y, z]` that describes the triplet you want to obtain.
To obtain target, you may apply the following operation on triplets any number of times (possibly zero):
- Choose two indices (0-indexed) `i` and `j` (`i` != `j`) and update `triplets[j]` to become `[max(ai, aj), max(bi, bj), max(ci, cj)]`.
- For example, if `triplets[i] = [2, 5, 3]` and `triplets[j] = [1, 7, 5]`, `triplets[j]` will be updated to `[max(2, 1), max(5, 7), max(3, 5)] = [2, 7, 5]`.
Return `true` if it is possible to obtain the target triplet `[x, y, z]` as an element of triplets, or `false` otherwise.

Example:
```
Input: triplets = [[2,5,3],[1,8,4],[1,7,5]], target = [2,7,5]
Output: true
Explanation: Perform the following operations:
- Choose the first and last triplets [[2,5,3],[1,8,4],[1,7,5]]. Update the last triplet to be [max(2,1), max(5,7), max(3,5)] = [2,7,5]. triplets = [[2,5,3],[1,8,4],[2,7,5]]
The target triplet [2,7,5] is now an element of triplets.
```

##### Intuition
The core idea is to determine if we can combine (merge) elements from different triplets to form the target triplet. For each element in the target triplet (x, y, z), we need at least one triplet whose corresponding element matches the target value or is less than or equal to it.

Key Points
- *Element-wise Check*: For each triplet, we check if any element exceeds the corresponding element in the target triplet. If any element is greater, that triplet cannot be used to form the target triplet.
- *Tracking Matches*: We use a set to keep track of which elements of the target triplet we have been able to match exactly (x, y, z). If, by the end of our iteration, we have matched all three elements, then we can form the target triplet.

Code
```python
def merge_triplets(triplets, target):  
    merge = set()  
    for triplet in triplets:  
        # Skip triplets with any element larger than the target's corresponding element  
        if triplet[0] > target[0] or triplet[1] > target[1] or triplet[2] > target[2]:  
            continue  
          
        # Check if any element in the triplet matches the corresponding target element  
        for index, value in enumerate(triplet):  
            if value == target[index]:  
                merge.add(index)  
      
    # Check if we have matched all three elements of the target triplet  
    return len(merge) == 3
```

#### Partition Labels
You are given a string `s`. We want to partition the string into as many parts as possible so that each letter appears in at most one part.
Note that the partition is done so that after concatenating all the parts in order, the resultant string should be `s`.
Return a list of integers representing the size of these parts.

Example:
```
Input: s = "ababcbacadefegdehijhklij"
Output: [9,7,8]
Explanation:
The partition is "ababcbaca", "defegde", "hijhklij".
This is a partition so that each letter appears in at most one part.
A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits s into less parts.
```

##### Intuition
To solve this problem, we need to keep track of the last occurrence of each character in the string. This will allow us to determine the boundaries of each partition. Here’s the step-by-step intuition:
- Track Last Occurrence: First, traverse the string to record the last occurrence of each character. This helps us know the furthest point each character needs to be included in a partition.
- Greedy Partitioning: Use two pointers to maintain the start and end of the current partition. As we traverse the string, we expand the end pointer to the maximum last occurrence of any character we encounter within the current partition. Once we reach the end of the current partition, we finalize it and start a new partition.
- Update Pointers: After finalizing a partition, update the start pointer to the next character and repeat the process until the entire string is partitioned.

```python
def partition_labels(s):
    # Record the last occurrence of each character
    last_occurence = {char: index for index, char in enumerate(s)}
    # Initialize pointers for partitioning  
    partitions = []  
    start = 0  
    end = 0 

    for index, char in enumerate(s):
        # Update the end pointer to the furthest last occurrence of the current character
        end = max(end, last_occurence[char])

        # If the current index matches the end pointer, finalize the partition
        if index == end:
            partitions.append(end-start+1)
            start = end + 1
    
    return partitions
```

#### Valid Parenthesis String
Given a string s containing only three types of characters: `'('`, `')'` and `'*'`, return `true` if `s` is valid.
The following rules define a valid string:
- Any left parenthesis '(' must have a corresponding right parenthesis ')'.
- Any right parenthesis ')' must have a corresponding left parenthesis '('.
- Left parenthesis '(' must go before the corresponding right parenthesis ')'.
- '*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string "".

Example 1:
```
Input: s = "()"
Output: true
```
Example 2:
```
Input: s = "(*)"
Output: true
```
Example 3:
```
Input: s = "(*))"
Output: true
```

##### Intuition
We can use a greedy algorithm with two variables to track the possible range of open parentheses counts:
- lo (low): The minimum possible number of open parentheses.
- hi (high): The maximum possible number of open parentheses.

As we traverse the string:
- If we encounter '(', we increment both lo and hi because it increases the count of open parentheses.
- If we encounter ')', we decrement both lo and hi because it decreases the count of open parentheses.
- If we encounter '*', we decrement lo (treating '*' as ')'), and increment hi (treating '*' as '(').

Key Points:
- Balance Tracking: lo should never be negative, because it represents the minimum possible open parentheses, and it can't drop below zero.
- Final Check: At the end of the traversal, lo should be zero, indicating that all open parentheses have been properly closed.

Code
```python
def check_valid_string(s):  
    lo = 0  # minimum open parentheses  
    hi = 0  # maximum open parentheses  
      
    for char in s:  
        if char == '(':  
            lo += 1  
            hi += 1  
        elif char == ')':  
            lo -= 1  
            hi -= 1  
        else:  # char == '*'  
            lo -= 1  # treat '*' as ')'  
            hi += 1  # treat '*' as '('  
          
        # If lo becomes negative, reset it to zero  
        lo = max(lo, 0)  
          
        # If hi is negative, it means there are too many ')' than '('  
        if hi < 0:  
            return False  
      
    # If lo is zero, it means all '(' have been matched with ')'  
    return lo == 0
```
