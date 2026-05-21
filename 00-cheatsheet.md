# Complete Algorithm Cheat Sheet

---

## 1. Prefix Sum

**Template:**
```python
prefix[i] = prefix[i-1] + arr[i]
range_sum(l, r) = prefix[r] - prefix[l-1]
```

| Problem | Approach | Trick |
|---------|----------|-------|
| [Range Sum Query](https://leetcode.com/problems/range-sum-query-immutable/) | Precompute prefix array | `prefix[r] - prefix[l-1]`; if l==0 return `prefix[r]` |
| [Subarray Sum = K](https://leetcode.com/problems/subarray-sum-equals-k/) | prefix + HashMap | If `(prefix_sum - k)` is in map → valid subarray found. Init map with `{0:1}` |

---

## 2. KMP

**LPS (Failure Function) template:**
```python
lps[0] = 0; j = 0
for i in 1..n:
    while j > 0 and s[i] != s[j]: j = lps[j-1]
    if s[i] == s[j]: j += 1
    lps[i] = j
```
**Search:** On mismatch at pattern[j], jump to `lps[j-1]` instead of restarting.
**Complexity:** O(n + m) time, O(m) space.

---

## 3. Sliding Window

**Fixed window:**
```python
for i in range(n):
    # add arr[i]; if i >= k: remove arr[i-k]; update answer
```

**Dynamic window:**
```python
left = 0
for right in range(n):
    # add arr[right] to state
    while window_invalid:
        # remove arr[left]; left += 1
    # update answer
```

**Before coding, write:** "This window is valid when ___"

| Problem | Window type | Key trick |
|---------|-------------|-----------|
| Max Sum Subarray of Size k | Fixed | Running sum; subtract outgoing element |
| First Negative in Window k | Fixed | Deque of indices; pop front when index out of window |
| [Count Anagram Occurrences](https://leetcode.com/problems/find-all-anagrams-in-a-string/) | Fixed | Compare char frequency maps |
| [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/) | Fixed | Monotonic decreasing deque; pop back if smaller than current |
| [Permutation in String](https://leetcode.com/problems/permutation-in-string/) | Fixed | freq maps equal → window is a permutation |
| Largest Subarray Sum = k | Dynamic | Shrink left when sum > k |
| [Longest Substring K Unique](https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/) | Dynamic | HashMap of char counts; shrink when unique > k |
| [Longest No Repeating Chars](https://leetcode.com/problems/longest-substring-without-repeating-characters/) | Dynamic | Set; shrink left until duplicate removed |
| [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/) | Dynamic | need_map vs have_map; expand until satisfied, then shrink |
| [Best Time to Buy/Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) | — | Track min_price so far; profit = price - min_price |
| [Longest Repeating Char Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/) | Dynamic | Valid if `(window_size - max_freq) <= k`; don't shrink max_freq on shrink |

---

## 4. Two Pointers

### Generic (opposite ends, sorted array)
```python
left, right = 0, n-1
while left < right:
    s = arr[left] + arr[right]
    if s == target: # found
    elif s < target: left += 1
    else: right -= 1
```

### Slow-Fast
```python
slow, fast = head, head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
```

**Floyd's cycle entrance:** after slow==fast, reset one to start, advance both by 1 → meeting point = cycle start.

| Problem | Type | Key trick |
|---------|------|-----------|
| [Two Sum (sorted)](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/) | Generic | Standard template above |
| [Three Sum](https://leetcode.com/problems/3sum/) | Generic | Fix one, two-pointer on rest. Skip dupes: `if i>0 and nums[i]==nums[i-1]: continue` |
| [Four Sum](https://leetcode.com/problems/4sum/) | Generic | Recursive: reduce to 2-sum. Fix 2 elements in nested loops |
| [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/) | Generic | `while left <= right` (same index valid for odd length) |
| [Container with Most Water](https://leetcode.com/problems/container-with-most-water/) | Generic | Move the shorter side (greedy — moving taller can only hurt) |
| [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/) | Generic | Move smaller side; `left_max - height[left]` gives water. **Update max before computing water** |
| Celebrity Problem | Generic | If left knows right → left eliminated. Else right eliminated. Then verify candidate in O(n) |
| [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/) | Slow-Fast | If no cycle, fast reaches null. If cycle, fast laps slow |
| [Middle of Linked List](https://leetcode.com/problems/middle-of-the-linked-list/) | Slow-Fast | When fast reaches end, slow is at middle |
| [Happy Number](https://leetcode.com/problems/happy-number/) | Slow-Fast | Model digit-square as implicit linked list; cycle = not happy |
| [Find Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/) | Slow-Fast | Array as implicit list: `i → arr[i]`. Floyd's → duplicate = cycle entrance |

**Duplicate skip after match (k-sum):**
```python
while left < right and nums[left] == nums[left-1]: left += 1
while left < right and nums[right] == nums[right+1]: right -= 1
```

---

## 5. Binary Search

**Always:** `mid = lo + (hi - lo) // 2`

| Pattern | Key invariant | Pointer update |
|---------|--------------|----------------|
| [Classic sorted](https://leetcode.com/problems/binary-search/) | standard | `lo=mid+1` or `hi=mid-1` |
| Descending | swap comparisons | same structure |
| [First/Last occurrence](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/) | don't stop on find | store mid, continue left/right |
| Count occurrences | first + last occurrence | `last - first + 1` |
| [Rotated: find min](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/) | min is where sorted breaks | if `mid < end` → go left, else go right |
| [Rotated: find target](https://leetcode.com/problems/search-in-rotated-sorted-array/) | one half always sorted | if target in sorted half → go there, else other half |
| Nearly sorted | check mid-1, mid, mid+1 | skip 2 positions on exclusion |
| Floor | store when `arr[mid] < val`, search right | `result = mid; lo = mid+1` |
| Ceil | store when `arr[mid] > val`, search left | `result = mid; hi = mid-1` |
| Infinite array | exponential search first | double hi until `arr[hi] >= target`, then BS |
| Min diff element | BS to insertion point | compare `arr[pos]` vs `arr[pos-1]` |
| [Time-Based KV Store](https://leetcode.com/problems/time-based-key-value-store/) | BS on timestamps per key | find largest timestamp `<= query` |
| [Median of 2 sorted](https://leetcode.com/problems/median-of-two-sorted-arrays/) | partition smaller array | `maxLeft1 <= minRight2 AND maxLeft2 <= minRight1` |
| [Peak element](https://leetcode.com/problems/find-peak-element/) | move toward greater neighbor | one side always has a peak |
| Bitonic max | = peak element | same logic |
| [Search in 2D matrix](https://leetcode.com/problems/search-a-2d-matrix/) | BS rows by first col, then BS row | O(log m + log n) |
| [**Allocate Books**](https://www.geeksforgeeks.org/allocate-minimum-number-pages/) | BS on answer: range = [max_book, total] | `isValid(mid)`: greedily allocate, check if `students <= m` |
| [Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/) | BS on speed: range = [1, max_pile] | `isValid(k)`: `sum(ceil(pile/k)) <= H` |
| Nth Root of M | BS on answer: range = [1, m] | `mid^n` compare with m |

---

## 6. Stack / Monotonic Stack

**NGR template (decreasing stack):**
```python
stack = []
for i in range(n):
    while stack and arr[stack[-1]] < arr[i]:
        result[stack.pop()] = arr[i]
    stack.append(i)
# remaining in stack → no greater element
```

| Variant | Stack | Direction |
|---------|-------|-----------|
| NGR (Next Greater Right) | Decreasing | L→R |
| NGL (Next Greater Left) | Decreasing | R→L |
| NSR (Next Smaller Right) | Increasing | L→R |
| NSL (Next Smaller Left) | Increasing | R→L |
| Stock Span | NGR/NGL | span = current_index - NGL_index |

| Problem | Approach | Key trick |
|---------|----------|-----------|
| Next Greater Element | Decreasing stack | Store indices; pop smaller on new larger element |
| Stock Span | NGR variant | Span = `i - stack[-1]` after popping smaller prices |
| [Max Area Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/) | NSL + NSR | `area = height[i] * (NSR[i] - NSL[i] - 1)` |
| [Trapping Rain Water (stack)](https://leetcode.com/problems/trapping-rain-water/) | Decreasing stack | Pop on taller bar; water = `(min(height[left], height[right]) - height[bottom]) * width` |
| [Min Stack (aux stack)](https://leetcode.com/problems/min-stack/) | Two stacks | Push to min_stack when `val <= min_stack[-1]` |
| Min Stack (O(1) space) | Encoded value | Push `2*val - min`; recover prev min = `2*min - stored` |
| [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/) | NGR | Store indices; answer = `i - popped_index` |
| [Car Fleet](https://leetcode.com/problems/car-fleet/) | Sort + stack | Sort by position desc; if time ≤ stack top → same fleet |
| Stack using Heap | Max-heap with counter | `(-counter, val)`; counter increments on each push |
| [Longest Valid Parentheses](https://leetcode.com/problems/longest-valid-parentheses/) | Stack of indices | Init with -1. For ')': pop; if empty → push i as base; else `len = i - stack[-1]` |
| [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/) | Stack | Map closing→opening; pop on close, check match |
| [Evaluate RPN](https://leetcode.com/problems/evaluate-reverse-polish-notation/) | Stack | Pop two on operator; mind order for `-` and `/` |
| Iterative Tower of Hanoi | 3 stacks | `2^n - 1` moves; pattern cycles every 3 moves |

---

## 7. Linked List

**Reversal dance:**
```python
prev, curr = None, head
while curr:
    nxt = curr.next; curr.next = prev; prev, curr = curr, nxt
return prev
```

| Problem | Approach | Key trick |
|---------|----------|-----------|
| [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/) | Prev/curr/next dance | Save next before reversing pointer |
| [Detect Cycle](https://leetcode.com/problems/linked-list-cycle/) | Slow-fast | Meet → cycle exists |
| [Find Cycle Start](https://leetcode.com/problems/linked-list-cycle-ii/) | Floyd's phase 2 | Reset one to head; advance both by 1 → meet at start |
| [Merge 2 Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/) | Dummy head + pointer | Compare and attach; append remaining |
| [Merge K Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/) | Pairwise merging | Merge 2 at a time log k rounds; O(n log k) |
| [Palindrome LL](https://leetcode.com/problems/palindrome-linked-list/) | Find mid + reverse half | Compare first half with reversed second half |
| [Copy with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/) | HashMap | First pass: create all nodes. Second pass: set next + random |
| [Remove Nth from End](https://leetcode.com/problems/remove-nth-node-from-end-of-list/) | Fast-slow gap of n | Advance fast by n first; then move both; slow.next is target |
| [**LRU Cache**](https://leetcode.com/problems/lru-cache/) | HashMap + Doubly LL | get/put O(1). Head = most recent. Tail = evict. Dummy head + tail |
| [Reverse k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/) | Reverse + connect | Track tail of prev group to connect to head of reversed group |
| [**Reorder List**](https://leetcode.com/problems/reorder-list/) | Mid + reverse + merge | Find mid → reverse second half → interleave |
| [Add Two Numbers](https://leetcode.com/problems/add-two-numbers/) | Traverse both + carry | Dummy head; handle remaining nodes and final carry |
| [Intersection](https://leetcode.com/problems/intersection-of-two-linked-lists/) | Length equalization | Advance longer by `|len1-len2|`; then advance together |

---

## 8. Trees / BST

**Core recursion question:** What from left child? What from right child? What to return to parent?

**Level Order BFS:**
```python
queue = deque([root])
while queue:
    for _ in range(len(queue)):  # len() gives current level size
        node = queue.popleft()
        # process; add children
```

| Problem | Approach | Key trick |
|---------|----------|-----------|
| [Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/) | BFS + queue | `len(queue)` before loop = level size |
| [Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/) | BFS, last per level | `if i == level_size - 1: result.append(node.val)` |
| [Populating Next Right Pointers](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/) | BFS | Connect nodes before enqueueing children |
| [Vertical Order Traversal](https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/) | BFS with col tracking | col-1 for left, col+1 for right; sort by (col, row, val) |
| Boundary Traversal | 3 passes | Left boundary (non-leaf) + all leaves + right boundary reversed |
| [Build Tree from Pre+Inorder](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) | Recursion | Preorder[0] = root; find in inorder to split left/right |
| [Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/) | Postorder | Swap children after recursing |
| [Max Depth](https://leetcode.com/problems/maximum-depth-of-binary-tree/) | Postorder | `1 + max(left, right)` |
| [Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/) | Postorder | Return `(is_balanced, height)`; early exit if unbalanced |
| [Same Tree](https://leetcode.com/problems/same-tree/) | Preorder | Check val + recurse both sides |
| [Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/) | Same tree check at each node | For each node, call isSameTree |
| [**LCA (Generic)**](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/) | Postorder | If both non-null returns, current = LCA. Else return the non-null |
| [Diameter](https://leetcode.com/problems/diameter-of-binary-tree/) | Postorder + global | `diameter = max(diameter, left_h + right_h)`; return `1 + max(left_h, right_h)` |
| [**Max Path Sum**](https://leetcode.com/problems/binary-tree-maximum-path-sum/) | Postorder + global | `gain = node.val + max(left,0) + max(right,0)`; update global; return `node.val + max(left,0,right,0)` |
| [Count Good Nodes](https://leetcode.com/problems/count-good-nodes-in-binary-tree/) | Preorder with max | Node is good if `val >= path_max`; pass updated max down |
| [**Serialize/Deserialize**](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/) | Preorder + null markers | Use iterator for deserialize; null = `#` |
| [Path Sum II](https://leetcode.com/problems/path-sum-ii/) | Backtracking DFS | `path.append → recurse → path.pop()` |
| [Flatten to LL](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/) | Postorder | Move left subtree to right; find tail of new right; attach old right |
| [**Validate BST**](https://leetcode.com/problems/validate-binary-search-tree/) | Preorder with (min,max) | Pass bounds down; left gets `(min, node.val)`, right gets `(node.val, max)` |
| [Sorted Array to BST](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/) | Divide and conquer | `mid = (lo+hi)//2` is root; recurse left and right halves |
| [LCA of BST](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/) | BST property | Both < root → go left; both > root → go right; else root is LCA |
| [Kth Smallest in BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/) | Inorder with counter | Stop at kth node; inorder = ascending order |
| [Inorder Successor](https://leetcode.com/problems/inorder-successor-in-bst/) | BST property | Has right child → leftmost in right. Else → nearest ancestor where node is in left subtree |
| [Closest Nodes Query](https://leetcode.com/problems/closest-nodes-queries-in-a-binary-search-tree/) | BST traversal | Track floor (largest ≤ query) and ceil (smallest ≥ query) during traversal |
| [BST Iterator](https://leetcode.com/problems/binary-search-tree-iterator/) | Stack simulation | Push all left children on init; `next()` pops, then pushes left children of right subtree |
| [Two Sum IV (BST)](https://leetcode.com/problems/two-sum-iv-input-is-a-bst/) | Two BST iterators | In-order ascending + reverse in-order descending → two-pointer |
| [Max Sum BST](https://leetcode.com/problems/maximum-sum-bst-in-binary-tree/) | Postorder | Return `(is_bst, sum, min, max)`; invalid propagates upward |

---

## 9. Heap

**Rule:** Kth largest → min-heap size k. Kth smallest → max-heap size k.
**Running median:** max-heap (lower half) + min-heap (upper half), sizes differ ≤ 1.

| Problem | Heap type | Key trick |
|---------|-----------|-----------|
| [Kth Largest in Array](https://leetcode.com/problems/kth-largest-element-in-an-array/) | Min-heap size k | Push; if size > k pop. Answer = `heap[0]` |
| Kth Smallest in Array | Max-heap size k | Negate values; same logic |
| Sort K-sorted Array | Min-heap size k+1 | k+1 guarantees min is in heap |
| K Closest Numbers | Max-heap size k | Store `(diff, val)`; negate diff for max-heap |
| [Top K Frequent](https://leetcode.com/problems/top-k-frequent-elements/) | Min-heap size k | Store `(count, val)` |
| Frequency Sort | Max-heap all | Negate counts; extract all in order |
| [K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/) | Max-heap size k | Store `(-dist, point)`; skip sqrt |
| Connect Ropes Min Cost | Min-heap | Always combine 2 shortest; push sum back |
| Sum Between K1 and K2 Smallest | Two heap queries | Find k1th and k2th smallest, sum elements between |
| [**Kth Largest in Stream**](https://leetcode.com/problems/kth-largest-element-in-a-stream/) | Min-heap size k | On add: push; if size > k pop; return `heap[0]` |
| [Last Stone Weight](https://leetcode.com/problems/last-stone-weight/) | Max-heap | Pop 2; push difference if non-zero |
| [**Task Scheduler**](https://leetcode.com/problems/task-scheduler/) | Max-heap + cooldown queue | Pop most frequent; add to cooldown queue with available time; re-add when time comes |
| [Design Twitter](https://leetcode.com/problems/design-twitter/) | Max-heap + timestamp | Collect tweets from user + followees; heap by `-timestamp` |
| [**Median from Stream**](https://leetcode.com/problems/find-median-from-data-stream/) | Two heaps | max-heap for lower, min-heap for upper; balance after each add |
| [K Pairs Smallest Sums](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/) | Min-heap | Init with `(nums1[i]+nums2[0], i, 0)` for first k; on pop push `(i, j+1)` |

---

## 10. Recursion

**IBH Method:** Hypothesis → Induction → Base case
**Input-Output (Recursive Tree):** Branches = choices. Leaf nodes = results. Draw tree first.

| Problem | Method | Key trick |
|---------|--------|-----------|
| Print 1 to N | IBH | Hypothesis: `f(n-1)` prints 1..n-1. Induction: print n after. Base: n==0 |
| Height of Binary Tree | IBH | Hypothesis: returns height of subtree. Induction: `1 + max(left, right)`. Base: None → 0 |
| Sort Stack | IBH | Hypothesis: sorts after removing top. Induction: insert top in correct position |
| Delete Middle of Stack | IBH | Helper tracks index; pop and push back around middle |
| Reverse Stack | IBH + helper | Pop all; `insert_at_bottom` puts each element at bottom |
| [Kth Symbol in Grammar](https://leetcode.com/problems/k-th-symbol-in-grammar/) | IBH | Row n has first half = row n-1; second half = complement. Reduce n |
| **Tower of Hanoi** | IBH | Move n-1 to helper; move nth to dest; move n-1 from helper to dest. **Roles shift each call** |
| Josephus Problem | IBH | Maintain persons list; `index = (index + k) % len`; pop; recurse |
| [**Subsets**](https://leetcode.com/problems/subsets/) | Input-Output | Include/exclude each element. Base: index == n → add current |
| [Subsets II (duplicates)](https://leetcode.com/problems/subsets-ii/) | Input-Output | Sort; when excluding, skip all dupes: `while nums[i]==nums[i+1]: i++` |
| Permutation with Spaces | Input-Output | For each char: recurse without space, recurse with space after |
| Permutation with Case Change | Input-Output | For each char: recurse with lower, recurse with upper |
| [**Balanced Parentheses**](https://leetcode.com/problems/generate-parentheses/) | Input-Output | Add `(` if open > 0; add `)` if close > open. Base: both == 0 |

---

## 11. Backtracking

**Blueprint:**
```python
def backtrack(state, start):
    if goal: result.append(state[:]); return
    for i in range(start, n):
        state.append(choices[i])       # make choice
        backtrack(state, next_start)   # recurse
        state.pop()                    # UNDO ← never forget
```

| Problem | Key detail | Duplicate handling |
|---------|-----------|-------------------|
| [Combination Sum](https://leetcode.com/problems/combination-sum/) | Same element reusable: `backtrack(state, i)` not `i+1` | Sort; skip `nums[i]==nums[i-1] when i>start` |
| [Combination Sum II](https://leetcode.com/problems/combination-sum-ii/) | Each element once: `backtrack(state, i+1)` | Sort; skip `nums[i]==nums[i-1] when i>start` |
| [Permutations](https://leetcode.com/problems/permutations/) | visited[] array or swap-in-place | — |
| [Permutations II](https://leetcode.com/problems/permutations-ii/) | Sort + visited; skip if `nums[i]==nums[i-1] and not visited[i-1]` | Sort required |
| Largest Number in K Swaps | Greedy backtrack: swap current with max in remaining | Track global max string |
| N-Digit Increasing Numbers | Next digit must be > previous | Start from prev_digit+1 |
| [Rat in a Maze](https://www.geeksforgeeks.org/rat-in-a-maze-backtracking-2/) | 4-directional DFS; mark as visited (set to 0); unmark on backtrack | — |
| [Word Search](https://leetcode.com/problems/word-search/) | 4-directional DFS; visited set per path | Unvisit on backtrack |
| [Palindrome Partitioning](https://leetcode.com/problems/palindrome-partitioning/) | For each start, try all substrings; if palindrome → recurse on rest | — |
| [Word Break (all sentences)](https://leetcode.com/problems/word-break-ii/) | Try all substring lengths; if in dict → recurse on rest | — |
| [Letter Combinations (Phone)](https://leetcode.com/problems/letter-combinations-of-a-phone-number/) | For each digit, try each letter; no condition needed | — |
| [**N-Queens**](https://leetcode.com/problems/n-queens/) | Track `cols`, `diag` (r-c), `anti_diag` (r+c) sets | Place one per row |
| [**Sudoku Solver**](https://leetcode.com/problems/sudoku-solver/) | Try 1-9 per empty cell; check row+col+box validity; reset on fail | Box index = `(r//3)*3 + c//3` |

---

## 12. Tries

```python
class TrieNode:
    def __init__(self): self.children = {}; self.is_end = False

# Insert: for ch in word: node = node.children.setdefault(ch, TrieNode()); node.is_end = True
# Search: for ch in word: if ch not in node.children: return False; ...return node.is_end
# StartsWith: same as search but return True at end (don't check is_end)
```

| Problem | Approach | Key trick |
|---------|----------|-----------|
| [Implement Trie](https://leetcode.com/problems/implement-trie-prefix-tree/) | Node with children dict + is_end | `setdefault` creates node if not present |
| [Longest Word All Prefixes](https://leetcode.com/problems/longest-word-in-dictionary/) | Insert all; DFS/BFS only through is_end nodes | Mark `is_prefix=True` on every insert node; verify all intermediate nodes are valid |
| [Word Search II](https://leetcode.com/problems/word-search-ii/) | Trie + grid DFS backtracking | Build trie from words; DFS from each cell; prune if char not in trie |

---

## 13. Greedy

**Trigger:** locally optimal choice provably leads to global optimum. Can't prove this? → DP instead.

| Problem | Sort by | Key trick |
|---------|---------|-----------|
| Activity Selection | End time | Pick earliest-ending activity; skip if it overlaps previous |
| Fractional Knapsack | val/weight ratio desc | Take full items with highest ratio; fraction of last |
| Job Sequencing | Profit desc | For each job, fill latest available slot ≤ deadline |
| Dijkstra's / Prim's | — | See Graph section |
| Kadane's / Max Product | — | See DP section |
| [Hand of Straights](https://leetcode.com/problems/hand-of-straights/) | — | Count freq; min-heap; form groups greedily from smallest |
| [Merge Triplets to Target](https://leetcode.com/problems/merge-triplets-to-form-target-triplet/) | — | Skip triplets exceeding target in any dimension; track which of 3 positions matched |
| [**Partition Labels**](https://leetcode.com/problems/partition-labels/) | — | `last_occurrence` map; extend end to `max(last[ch])`; finalize partition at end |
| [Valid Parenthesis String](https://leetcode.com/problems/valid-parenthesis-string/) | — | Track `(lo, hi)` range of possible open counts. `*` → `(lo-1, hi+1)`. If hi < 0 → invalid. Clamp lo ≥ 0. Valid if lo == 0 at end |

---

## 14. Intervals

**Almost always: sort first.**

| Problem | Sort by | Merge condition | Key trick |
|---------|---------|-----------------|-----------|
| [Merge Intervals](https://leetcode.com/problems/merge-intervals/) | Start | `curr_start <= prev_end` | Extend: `prev_end = max(prev_end, curr_end)` |
| [Non-Overlapping (min removals)](https://leetcode.com/problems/non-overlapping-intervals/) | End | If overlap → remove current (keep earlier-ending) | Count removed intervals |
| [Meeting Rooms](https://leetcode.com/problems/meeting-rooms/) | Start | Any overlap → False | Check `intervals[i][0] < intervals[i-1][1]` |
| [**Meeting Rooms II**](https://leetcode.com/problems/meeting-rooms-ii/) | — | — | Separate starts/ends arrays, sort both. Two pointers: rooms++ on start, rooms-- on end |
| [**Min Interval per Query**](https://leetcode.com/problems/minimum-interval-to-include-each-query/) | Sort intervals by size | — | Sort queries; add intervals with start ≤ query to min-heap (size-keyed); remove expired; top = answer |

---

## 15. Graphs

**BFS template:** queue + visited set. Level = distance.
**DFS template:** recursive or stack + visited set.

**Cycle detection:**
- Undirected: BFS/DFS with parent tracking. Visited neighbor ≠ parent → cycle.
- Directed: DFS with `WHITE/GRAY/BLACK` states. GRAY on revisit → cycle.

**Topological Sort (Kahn's BFS):**
```python
in_degree = [0] * V
# fill in_degree from edges
queue = [v for v in range(V) if in_degree[v] == 0]
while queue:
    node = queue.pop(); order.append(node)
    for nb in graph[node]:
        in_degree[nb] -= 1
        if in_degree[nb] == 0: queue.append(nb)
# if len(order) < V → cycle
```

**Topological Sort (DFS):** postorder append + reverse. Track WHITE/GRAY/BLACK.

**Union-Find:**
```python
def find(x):  # path compression
    if parent[x] != x: parent[x] = find(parent[x])
    return parent[x]
def union(x, y):  # by rank
    px, py = find(x), find(y)
    if rank[px] < rank[py]: px, py = py, px
    parent[py] = px
    if rank[px] == rank[py]: rank[px] += 1
```

**Dijkstra:**
```python
dist = [inf] * V; dist[src] = 0
heap = [(0, src)]
while heap:
    d, u = heappop(heap)
    if d > dist[u]: continue   # stale entry
    for v, w in graph[u]:
        if dist[u] + w < dist[v]:
            dist[v] = dist[u] + w; heappush(heap, (dist[v], v))
```

| Problem | Algorithm | Key trick |
|---------|-----------|-----------|
| [Number of Provinces](https://leetcode.com/problems/number-of-provinces/) | DFS/BFS | Count DFS initiations on unvisited nodes |
| [**Number of Islands**](https://leetcode.com/problems/number-of-islands/) | DFS | Modify grid in-place (1→0) as visited marker |
| [Max Area of Island](https://leetcode.com/problems/max-area-of-island/) | DFS | Return cell count from DFS |
| [Clone Graph](https://leetcode.com/problems/clone-graph/) | DFS + HashMap | Map original→clone; check map before creating |
| [Word Ladder](https://leetcode.com/problems/word-ladder/) | BFS + pattern | `*` wildcard patterns group similar words; BFS = shortest path |
| Cycle (directed) | DFS + recur stack | GRAY on revisit = back edge = cycle |
| Cycle (undirected) | BFS with parent | Visited neighbor ≠ parent → cycle |
| [Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/) | DFS cycle + connectivity | No cycle AND all n nodes visited |
| [**Topological Sort**](https://www.geeksforgeeks.org/topological-sorting/) | Kahn's or DFS | If processed count < V → cycle exists |
| [**Course Schedule I**](https://leetcode.com/problems/course-schedule/) | Topo sort | Cycle exists → can't finish all courses |
| [**Course Schedule II**](https://leetcode.com/problems/course-schedule-ii/) | Topo sort | Return ordering; empty if cycle |
| [Alien Dictionary](https://leetcode.com/problems/alien-dictionary/) | Build graph + topo | Compare adjacent words; first diff char → edge |
| [Rotten Oranges](https://leetcode.com/problems/rotting-oranges/) | Multi-source BFS | Init queue with all rotten; levels = minutes |
| [Pacific Atlantic](https://leetcode.com/problems/pacific-atlantic-water-flow/) | Reverse DFS from edges | DFS from each ocean's edges uphill; intersection = answer |
| [Surrounded Regions](https://leetcode.com/problems/surrounded-regions/) | DFS from border O's | Mark border-connected O's as T; flip O→X, T→O |
| [Walls and Gates](https://leetcode.com/problems/walls-and-gates/) | Multi-source BFS | Init with all gates; BFS updates INF cells with distance |
| [**Dijkstra**](https://leetcode.com/problems/network-delay-time/) | Min-heap + dist[] | Skip stale entries `if d > dist[u]: continue` |
| [Network Delay Time](https://leetcode.com/problems/network-delay-time/) | Dijkstra | Answer = max of all shortest distances; -1 if unreachable |
| [Swim in Rising Water](https://leetcode.com/problems/swim-in-rising-water/) | Dijkstra variant | Cost = max elevation seen (not sum) |
| [Path with Min Effort](https://leetcode.com/problems/path-with-minimum-effort/) | Dijkstra variant | Cost = max abs height diff seen |
| Bellman-Ford | V-1 relaxations | Handles negative weights; Vth relaxation → negative cycle |
| [Cheapest Flights K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/) | Bellman-Ford limited | Only k+1 relaxation iterations |
| **Floyd-Warshall** | Triple loop | `dist[i][j] = min(dist[i][j], dist[i][k]+dist[k][j])` for all k |
| [Find City (threshold)](https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/) | Floyd-Warshall | Count reachable cities within threshold; return city with fewest (highest index on tie) |
| **Union-Find** | path compression + rank | `find` with compression; `union` by rank |
| [Connected Components](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/) | Union-Find | Count unique roots after processing all edges |
| [Redundant Connection](https://leetcode.com/problems/redundant-connection/) | Union-Find | First edge where both nodes already share root = redundant |
| [Accounts Merge](https://leetcode.com/problems/accounts-merge/) | Union-Find | Union all emails in an account; group by root |

---

## 16. Dynamic Programming

**Protocol:** Define `dp[i]` precisely in English → recurrence → base cases → top-down or bottom-up

**Greedy vs DP:** If locally optimal is provably safe → greedy. If current choice depends on future → DP.

### 0-1 Knapsack pattern
```python
dp[i][w] = max(dp[i-1][w], val[i] + dp[i-1][w-wt[i]])  # include or exclude
```
| Problem | Variation |
|---------|-----------|
| 0-1 Knapsack | Classic |
| Subset Sum | Target = capacity; return bool |
| [Equal Sum Partition](https://leetcode.com/problems/partition-equal-subset-sum/) | Subset sum with target = total/2 |
| Count Subsets with Sum | Return count; skip dupes when excluding |
| Min Subset Sum Difference | Find all possible sums up to total/2; pick closest to total/2 |
| Number of Subsets with Diff d | `S1 = (total + d) / 2`; count subsets with that sum |
| [Target Sum (+/-)](https://leetcode.com/problems/target-sum/) | Same as above: `S1 = (total + target) / 2` |

### Unbounded Knapsack pattern
```python
dp[w] = opt(dp[w], dp[w - coin] + 1)   # inner loop doesn't reset; reuse allowed
```
| Problem | Variation |
|---------|-----------|
| Unbounded Knapsack | Max value, unlimited items |
| Rod Cutting | Lengths = weights, prices = values |
| [Coin Change (min coins)](https://leetcode.com/problems/coin-change/) | Minimize count |
| [Coin Change II (ways)](https://leetcode.com/problems/coin-change-2/) | Count combinations (coins in outer loop) |
| [Word Break](https://leetcode.com/problems/word-break/) | dp[i] = True if s[:i] is segmentable; try all j < i where s[j:i] in dict |

### LCS / String DP pattern
```python
dp[i][j] = dp[i-1][j-1] + 1 if match else max(dp[i-1][j], dp[i][j-1])
```
| Problem | Key modification |
|---------|-----------------|
| LCS | Standard |
| Print LCS | Backtrack dp table: diagonal on match, else larger direction |
| Longest Common Substring | `dp[i][j] = 0 on mismatch`; track max |
| Shortest Common Supersequence | Length = m + n - LCS |
| Longest Palindromic Subsequence | LCS(s, reverse(s)) |
| Min Deletions for Palindrome | n - LPS |
| Min Insertions for Palindrome | n - LPS |
| Longest Repeating Subsequence | LCS(s, s) but `i ≠ j` on match |
| [Interleaving String](https://leetcode.com/problems/interleaving-string/) | dp[i][j]: can s3[:i+j] be formed by interleaving s1[:i] and s2[:j] |
| [Distinct Subsequences](https://leetcode.com/problems/distinct-subsequences/) | Count matches; `dp[i][j] = dp[i+1][j+1] + dp[i+1][j]` on match |
| [Edit Distance](https://leetcode.com/problems/edit-distance/) | `dp[i][j] = dp[i-1][j-1]` on match; else `1 + min(del, ins, rep)` |

### LIS pattern
```python
dp[i] = 1 + max(dp[j] for j < i if arr[j] < arr[i])  # O(n²)
# O(n log n): patience sort with binary search
```
| Problem | Key |
|---------|-----|
| [LIS](https://leetcode.com/problems/longest-increasing-subsequence/) | Standard |

### Kadane's / Product
```python
dp[i] = max(arr[i], dp[i-1] + arr[i])          # max subarray sum
# for product: track max and min (negatives flip sign)
```
| Problem | Key |
|---------|-----|
| [Max Subarray](https://leetcode.com/problems/maximum-subarray/) | Standard Kadane's |
| [Max Product Subarray](https://leetcode.com/problems/maximum-product-subarray/) | Track `(max_prod, min_prod)`; swap on negative element |
| [Max Alternating Subsequence Sum](https://leetcode.com/problems/maximum-alternating-subsequence-sum/) | Two states: `(last_positive, last_negative)` |
| [Decode Ways](https://leetcode.com/problems/decode-ways/) | dp[i] = ways to decode s[:i]; check 1-digit and 2-digit codes |

### Fibonacci / Linear DP
```
dp[i] = dp[i-1] + dp[i-2]
```
| Problem | Formula |
|---------|---------|
| [Fibonacci](https://leetcode.com/problems/fibonacci-number/) | `F(n) = F(n-1) + F(n-2)` |
| [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/) | Same as Fibonacci |
| [Min Cost Climbing Stairs](https://leetcode.com/problems/min-cost-climbing-stairs/) | `dp[i] = cost[i] + min(dp[i-1], dp[i-2])`; answer = `min(dp[-1], dp[-2])` |
| [House Robber](https://leetcode.com/problems/house-robber/) | `dp[i] = max(dp[i-1], nums[i] + dp[i-2])` |

### MCM / Interval DP pattern
```python
dp[i][j] = min(dp[i][k] + dp[k+1][j] + cost(i,k,j)) for k in [i, j-1]
```
| Problem | Variation |
|---------|-----------|
| Matrix Chain Multiplication | cost = `arr[i-1]*arr[k]*arr[j]` |
| [Palindrome Partitioning (min cuts)](https://leetcode.com/problems/palindrome-partitioning-ii/) | If palindrome: 0. Else: `1 + min cuts over all splits` |
| Boolean Parenthesization | Count True/False combinations with each operator |
| [Scramble String](https://leetcode.com/problems/scramble-string/) | Split + swap/no-swap check with memoization |
| Egg Drop | dp[e][f] = trials; try each floor k; worst case of break/no-break |
| [Burst Balloons](https://leetcode.com/problems/burst-balloons/) | Burst k last in [l,r]; `coins = nums[l-1]*nums[k]*nums[r+1] + dp[l][k-1] + dp[k+1][r]` |

### Grid DP
```python
dp[i][j] = dp[i-1][j] + dp[i][j-1]             # unique paths
dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])  # min path sum
dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])  # maximal square
```
| Problem | Formula |
|---------|---------|
| [Unique Paths](https://leetcode.com/problems/unique-paths/) | Sum from top and left |
| [Unique Paths with Obstacles](https://leetcode.com/problems/unique-paths-ii/) | `dp[i][j] = 0` if obstacle |
| [Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/) | `grid[i][j] + min(top, left)` |
| [Longest Increasing Path in Matrix](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/) | DFS + memo; 4-directional, strictly increasing |
| [Maximal Square](https://leetcode.com/problems/maximal-square/) | `1 + min(top, left, diag)` if cell is '1' |

### Palindromic DP
```python
# Expand from center (O(n²) time, O(1) space)
for each center (char and pair):
    expand left/right while match; count/track palindromes
```
| Problem | Key |
|---------|-----|
| [Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/) | Count all; expand from center |
| Longest Palindromic Subsequence | LCS(s, reverse(s)) |

### DP on Trees
Every tree DP is postorder: get values from children, compute for current, return to parent.
| Problem | Return | Global update |
|---------|--------|---------------|
| [Diameter](https://leetcode.com/problems/diameter-of-binary-tree/) | height | `max(diam, left_h + right_h)` |
| Max Path Sum (leaf to leaf) | best path to descendant | `max(res, node + left + right)` |
| [Max Path Sum (any to any)](https://leetcode.com/problems/binary-tree-maximum-path-sum/) | `node + max(0, left, right)` | `max(res, node + max(0,left) + max(0,right))` |

### Catalan Numbers
```python
C(n) = sum(C(i) * C(n-1-i) for i in 0..n-1);  C(0) = C(1) = 1
```
| Problem | n |
|---------|---|
| [Unique BSTs with n nodes](https://leetcode.com/problems/unique-binary-search-trees/) | n keys |
| [Generate Parentheses](https://leetcode.com/problems/generate-parentheses/) | n pairs |

---

## Master Trigger Reference

| Symptom | Pattern |
|---------|---------|
| "contiguous subarray with condition" | Sliding Window |
| "sorted array, find pair/triple" | Two Pointers |
| "eliminate half search space" | Binary Search |
| "next greater/smaller element" | Monotonic Stack |
| "repeatedly find extreme element" | Heap |
| "path in graph, connected components" | BFS/DFS |
| "no cycles + ordering of dependencies" | Topological Sort |
| "shortest path (no neg weights)" | Dijkstra |
| "shortest path (neg weights)" | Bellman-Ford |
| "all pairs shortest path" | Floyd-Warshall |
| "merge/group connected sets" | Union-Find |
| "prefix matching, multiple words" | Trie |
| "locally optimal → global optimal" | Greedy |
| "overlapping subproblems + optimal substructure" | DP |
| "all combinations/permutations + constraints" | Backtracking |
| "tree problem: what from children?" | IBH Recursion |
