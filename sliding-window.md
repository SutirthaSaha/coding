# Sliding Window
Problem:
Given an array and a number 'k', return the maxinum sum of all possible subarrays of size 'k'.
Naive Solution:
We can start with each index and loop for 'k' elements and calculate the sum for each index and calculate the maximum.
```python
def find_max_sum_subarray(nums, k):
    n = len(nums)
    max_sum = float('inf')
    for i in range(n - k + 1):
        curr_sum = 0
        for j in range(i, i+3):
            curr_sum = curr_sum + nums[j]
        max_sum = max(max_sum, curr_sum)
    return max_sum
```
This is an `O(n*k)` approach, can we make it better?

## Identification
Check for 3 things:
- input is an array or string
- there is a window size provided or asks for a window size satisfying a particular condition (largest/smallest/unique)
- are we doing any repetitive work?

## Types
There are 2 types of sliding window problem:
- **Fixed Window Size**: The window size is provided as input
- **Dynamic Window Size**: The problem statement requires to find the largest/smallest satisfying a particular condition

Now how do we solve the previous problem. The problem has window size provided as input - fixed sliding window problem.

```python
def find_max_sum_subarray(nums, k):
    n = len(nums)
    max_sum = float('-inf')
    curr_sum = 0
    start = 0
    for end in range(n):
        # perform operations for each element being added to the window
        curr_sum = curr_sum + nums[end]

        # check if the window is in limits
        # revert the changes for the starting element of the window
        if (end - start + 1) > k:
            curr_sum = curr_sum - nums[start]
            start = start + 1
        
        # if the window size if mathcing, add to the result
        if (end - start + 1) == k:
            max_sum = max(max_sum, curr_sum)
    return max_sum
```
This would be an O(n) solution and we would use the sum calculated in the previous window - thus reducing repetitive calculation.

## Problems
### First -ve number in each window of size k
We can store each -ve number we encounter in a list or queue and whenever we exceed the window size we can pop from the left if the index is equal to element index being removed from the window.

```python
def first_negative(nums, k):
    queue = deque()
    result = []
    start = 0
    for end in range(n):
        if nums[end] < 0:
            queue.append(end)
        if (end - start + 1) > k:
            if start == queue[0]:
                queue.popleft()
            start = start + 1
        if (end - start + 1) == k:
            if queue:
                result.append(nums[queue[0]])
            else:
                result.append(0)
    return result 
```

### Count Occurences of Anagrams
Given a input string and a smaller string, we need to find the number of anagrams for the smaller string.
**Anagrams**: Two string having the same characters in the same count. 

```python
from collections import defaultdict

def count_anagrams(string, sub_string):
    counter = defaultdict(int)
    start = 0
    count = 0

    match_counter = defaultdict(int)
    for char in sub_string:
        match_counter[char] = match_counter[char] + 1
    
    def is_anagram(match_counter, counter):
        for char in match_counter:
            if char not in counter:
                return False
            if match_counter[char] != counter[char]:
                return False
        return True

    for end in range(len(string)):
        counter[string[end]] = counter[string[end]] + 1
        if (end - start + 1) > k:
            counter[string[start]] = counter[string[start]] - 1
            start = start + 1
        if (end - start + 1) == k:
            if is_anagram(match_counter, counter):
                count = count + 1
        return count         
```
Always remember for each new index you encounter - you perform some work. Whenever the window size exceeds the limit, you revert the changes.

### Maximum of all subarrays of size k*
Given an array and a size of window k, we need to find out all the maximums in a window of size k.

### Intuition
For this we need a data structure that has efficient operations on both ends - double ended queue, and we would store the possible contenders for the maximum value in it. So we pop if we encounter any value greater than it in the window.
- this ensures that we have already the next maximum with us when we are removing elements that are not in the window but was also a contender for the maximum.

```python
def subarray_max(nums, k):
    result = []
    start = 0
    max_list = deque()

    for end in range(len(nums)):
        while nums[max_list[-1]] <= nums[end]:
            max_list.pop()
        max_list.append(end)

        if (end - start + 1) > k:
            if start == max_list[0]:
                max_list.popleft()
            start = start + 1
        
        if (end - start + 1) == k:
            result.append(nums[max_list[-1]])
    return result
```

Now we'll look at some problems of variable size sliding window.

### Largest Subarray or sum k
Here as you can see that we would have to maximize the window size having a sum of k (matching a particular condition).
Here the window size is not specified but a condition is specified. This is a dynamic/variable size sliding window problem.

```python
def larget_subarray(nums, k):
    max_start, max_end = 1, 0
    start = 0
    curr_sum = 0

    for end in range(len(nums)):
        # calculation
        curr_sum = curr_sum + nums[end]

        # while the condition is invalid, reduce the window size
        while curr_sum > k:
            curr_sum = curr_sum - nums[start]
            start = start + 1
        
        # if the condition matches, update the max window is condition matches
        if curr_sum == k:
            curr_len = end - start + 1
            max_len = max_end - max_start + 1
            if curr_len > max_len:
                max_start, max_end = start, end
    return nums[max_start: max_end+1]
```
In the code you can notice that now the conditions are not based on the window size but on the conditions provided by the problem statement.

### Longest Substring with K unique characters
Input is a string and a number k. We would have to find the longest substring with `k` unique characters.
```python
def longest_substring_k_unique(string, k):
    result = 0
    def count_unique(counter):
        count = 0
        for char in counter:
            if counter[char] > 0:
                count = count + 1
        return count
    counter = defaultdict(int)
    start = 0
    for end in range(len(string)):
        counter[string[end]] = counter[string[end]] + 1
        while count_unique(counter) > k:
            counter[string[start]] = counter[string[start]] - 1
            start = start + 1
        if count_unique(counter) == k:
            result = max(result, end - start + 1)
    return result
```

### Longest Substring Without any Repeating Characters*
Input is a string and we need to provide the longest substring with no repeating characters.
```python
def longest_substring(string):
    char_set = set()
    result = 0
    start = 0

    for end in range(len(string)):
        char = string[end]

        # If the character is already in the set, shrink the window from the left
        while char in char_set:
            char_set.remove(string[start])
            start = start + 1
        
        char_set.add(char)
        result = max(result, end - start + 1)
    return result
```

### Minimum Window Substring*
Given two strings `s` and `t` of lengths `m` and `n` respectively, return the minimum window 
substring of `s` such that every character in `t` (including duplicates) is included in the window.

#### Intuition
The idea is to dynamically adjust the window size to find the smallest substring that contains all characters from the target string t.
- Create a map for the frequency of each character in t `Need Map` and another map to track the frequency of characters within the current window of s `Have Map`.
- Increase the window size, adding characters to the `Have Map` until the window contains all required characters from the `Need Map`.
- Once the window is sufficient (contains all characters from t), contract the window from left and try to minimize its size while still containing all required characters.
- Keep track of the smallest valid window found.

Code
```python
def minimum_window(s, t):
    # Helper function to check if the current window is sufficient
    def check_sufficient():
        for char in need_map:
            if need_map[char] > have_map[char]:
                return False
        return True

    m, n = len(s), len(t)
    min_start, min_end = 0, m-1
    start = 0
    have_map, need_map = defaultdict(int), defaultdict(int)

    # Populate the need_map with the frequency of each character in t
    for char in t:
        need_map[char] = need_map[char] + 1
    
    # Flag to indicate if a valid window has been found
    is_sufficient = False

    for end in range(m):
        have_map[s[end]] = have_map[s[end]] + 1

        # Check if the current window is sufficient and try to contract it
        while check_sufficient():
            is_sufficient = True
            curr_window = end - start + 1

            # Update the minimum window if the current one is smaller
            if (min_end - min_start + 1) > curr_window:
                min_start, min_end = start, end

                # Contract the window from the left
                have_map[s[start]] = have_map[s[start]] - 1
                start = start + 1
    
    # If a valid window was found, return the smallest window substring
    if is_sufficient:
        return s[min_start: min_end+1]
    return ""
```

### Best Time to Buy And Sell Stock*
You are given an array prices where prices[i] is the price of a given stock on the ith day.
You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

Example 1:
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
Example 2:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.

#### Intuition
To solve this problem efficiently, we can use a single pass approach. The key is to keep track of the minimum price encountered so far as we iterate through the array and calculate the potential profit at each step.

Code
```python
def max_profit(prices):
    min_price = float('inf')
    max_profit = 0

    for price in prices:
        # Update the minimum price encountered so far - this would be the buy price
        if price < min_price:
            min_price = price
        
        # Calculate the potential profit for the current price
        profit = price - min_price

        # Update the maximum profit if the current profit is higher
        max_profit = max(max_profit, profit)
    
    return max_profit
```

### Permutation in String*
Given two strings s1 and s2, return true if s2 contains a 
permutation of s1, or false otherwise.
In other words, return true if one of s1's permutations is the substring of s2.

Example 1:
Input: s1 = "ab", s2 = "eidbaooo"
Output: true
Explanation: s2 contains one permutation of s1 ("ba").

Example 2:
Input: s1 = "ab", s2 = "eidboaoo"
Output: false

#### Intuition:
- Two strings are permutations of each other if and only if they have the same character frequencies. For example, both "ab" and "ba" contain one 'a' and one 'b'.
- **A sliding window of fixed size** (length of s1) can be used to traverse s2. This allows us to check each substring of s2 of the same length as s1 for matching character frequencies.

```python
def permutation(s1, s2):
    n1, n2 = len(s1), len(s2)
    count1, count2 = defaultdict(int), defaultdict(int)

    # Count characters in s1
    for char in s1:
        count1[char] = count1[char] + 1
    
    # Sliding window initialization
    start = 0
    for end in range(n2):
        char = s2[end]
        count2[char] = count2[char] + 1

        # Ensure the window size does not exceed the length of s1
        if (end - start + 1) > n1:
            count2[s2[start]] = count2[s2[start]] - 1
            if count2[s2[start]] == 0:
                del count2[s2[start]]
            start = start + 1
        
        # Check if the current window matches the character counts of s1
        if (end - start + 1) == n1:
            if count1 == count2:
                return True
    
    return False
```

### Sliding Window Maximum* - Already solved above as Maximum of Subarrays of Size k
You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position.

Return the max sliding window.
Example 1:
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3  
Output: [3,3,5,5,6,7]  
  
Explanation:
```  
Window position                Max  
---------------               -----  
[1  3  -1] -3  5  3  6  7       3  
 1 [3  -1  -3] 5  3  6  7       3  
 1  3 [-1  -3  5] 3  6  7       5  
 1  3  -1 [-3  5  3] 6  7       5  
 1  3  -1  -3 [5  3  6] 7       6  
 1  3  -1  -3  5 [3  6  7]      7  
```

Example 2:
Input: nums = [1], k = 1
Output: [1]

#### Intuition
- **Fixed Sliding Window** - The problem requires finding the maximum value in each sliding window of size `k` as it moves across the array from left to right.
- **Deque** - We use a deque to store the indices of elements in such a way that the largest element for the current window is always at the front. This allows efficient access to the maximum value.
- For maintaining the deque we have 2 major operations:
  - Pop all the elements in deque from the end which are lesser than the current element.
  - If the front element (the maximum one) is out of the window - pop it out.

```python
def sliding_window_maximum(nums):
    n = len(nums)
    queue = deque()
    result = []

    start = 0
    for end in range(n):
        # Remove elements from the front of the queue that are out of the current window
        if queue and queue[0] < start:
            queue.popleft()
        
        # Remove elements from the back of the queue that are smaller than the current element
        while queue and nums[queue[-1]] < nums[end]:
            queue.pop()
        
        # Add the current element index to the queue
        queue.append(end)

        # Once the window is fully within the bounds of the array (i.e., end >= k - 1)
        if end >= k-1:
            # Append the max value for the current window to the result
            result.append(nums[queue[0]])
            # Move the start of the window
            start = start + 1
    
    return result
```

### Longest Repeating Character Replacement*
You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times.

Return the length of the longest substring containing the same letter you can get after performing the above operations.

Example 1:
Input: s = "ABAB", k = 2
Output: 4
Explanation: Replace the two 'A's with two 'B's or vice versa.

Example 2:
Input: s = "AABABBA", k = 1
Output: 4
Explanation: Replace the one 'A' in the middle with 'B' and form "AABBBBA". The substring "BBBB" has the longest repeating letters, which is 4. There may exists other ways to achieve this answer too.

#### Intuition
- **Dynamic Sliding Window**- Here the sliding window has to be adjusted to ensure the condition of maximum `k` replacements is always satisfied.
- We would maintain `max_freq` to keep track of the character that appears the most within the window. This helps us understand how many characters we need to change - when this exceeds `k` we shrunk the window. Although we don't update the `max_freq` on shrinking as if the later substrings don't exceed this value - it would never be a valid candidate - it acts as an upper bound.
- We also maintain the count of the characters, when expanding we check with the count of the current character if it can be the character with `max_freq` or not. On shrinking the window we reduce the count of the character at start of the window.

```python
def longest_repeating_character_replacement(s, k):
    n = len(s)
    max_freq = 0
    max_length = 0
    count = defaultdict(int)

    start = 0
    for end in range(n):
        # Increment the frequency of the current character
        count[s[end]] = count[s[end]] + 1

        # Update the maximum frequency of a single character in the current window
        max_freq = max(max_freq, count[s[end]])

        # If the number of characters to change exceeds k, shrink the window from the left by 1
        if (end - start + 1 - max_freq) > k:
            count[s[start]] = count[s[start]] - 1
            start = start + 1
        
        # Update the maximum length of the substring found so far 
        max_length = max(max_length, end-start+1)
    
    return max_length
```