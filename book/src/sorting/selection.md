# Selection and equivalence classes

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Find the $m$ largest elements without sorting everything, and group items into the classes of an equivalence relation |
| **Devised by** | J. W. J. Williams (1964), heaps; Bernard Galler and Michael Fischer (1964), union-find; the bounded-heap selection is folklore |
| **Accuracy** | Exact; results are checked against a full sort in the generator |
| **Cost** | Selection $O(n \log m)$ time and $O(m)$ memory; equivalence classes $O(n^2)$ relation tests |
| **API** | [`select_m_largest`](../../api/lib_jsl_sorting/select_m_largest/fn.select_m_largest.html), [`equivalence_classes`](../../api/lib_jsl_sorting/equivalence_classes/fn.equivalence_classes.html) |

</div>

## Overview

Often only part of a sorted order is wanted: the ten best scores, the hundred nearest candidates, the largest peaks of a spectrum. Sorting everything costs $O(n \log n)$. Keeping the best $m$ seen so far in a small heap costs $O(n \log m)$, reads the data once, and needs only $O(m)$ memory, so it works on streams too long to hold.

Another frequent need is grouping: which files are duplicates, which words are anagrams, which measurements belong to the same object. When "belongs with" is an **equivalence relation**, reflexive, symmetric and transitive, the items fall into disjoint **equivalence classes**. If the given relation is not transitive, for example "within 2 of each other", the library takes its transitive closure: two items are in the same class when a chain of related items joins them.

## History

Heaps were introduced by J. W. J. Williams in 1964 for heapsort and soon became the standard priority queue. Selecting the $k$ smallest with a bounded heap appears in early textbooks on sorting and in database "top-$k$" query processing. The exact $O(n)$ selection algorithms are Hoare's quickselect (1961) and the median-of-medians method of Blum, Floyd, Pratt, Rivest and Tarjan (1973). Equivalence classes were the original application of the disjoint-set forest of Bernard Galler and Michael Fischer (1964), devised for FORTRAN's `EQUIVALENCE` statements. Numerical Recipes popularised a routine of the same name.

## The idea

### Selecting the m largest

Keep a **min-heap** of at most $m$ elements. Its root is the smallest of the best $m$ so far, the threshold for getting in. For each new element: if the heap is not full, add it; otherwise, if the element beats the root, replace the root. At the end, sort the $m$ survivors in descending order. Each step costs at most $O(\log m)$, and for random input most elements fail the threshold test after one comparison.

```rust,ignore
{{#include ../../../crates/sorting/src/select_m_largest.rs:select}}
```

{{#include ../generated/sorting/selection.md}}

For small $m$ selection is over twenty times faster than sorting. As $m$ approaches $n$ it loses that advantage. The heap then does the work of a heapsort, and the standard library's sort is faster.

### Equivalence classes

Each item starts in its own class. For every pair $(i, j)$ not yet in the same class, test the relation, and if it holds, merge their classes with a [union-find](../graph/spanning-trees.md#union-find) structure. Path halving keeps the trees shallow. At the end, items are gathered by their class representative. Classes come out in order of their first member, with members in input order.

```rust,ignore
{{#include ../../../crates/sorting/src/equivalence_classes.rs:classes}}
```

Testing every pair costs $O(n^2)$ relation tests:

{{#include ../generated/sorting/equivalence.md}}

That suits relations given only as a yes/no test. When items have a **canonical key**, such as the sorted letters of a word or a rounded value, sorting or hashing by the key groups them in $O(n \log n)$ or $O(n)$ instead.

## Algorithm

1. **Selection**: bounded min-heap of size $m$; replace the root when a larger element arrives; sort the heap's contents descending.
2. **Equivalence classes**: union-find over indices; for each pair in different sets, test and union (smaller index as root); gather by root.

## Implementation

- `select_m_largest` clones the selected elements (`T: Ord + Clone`) and returns them largest first; `m = 0` gives an empty vector and $m \ge n$ all elements sorted.
- `equivalence_classes` takes any closure `Fn(&T, &T) -> bool`, which need only be symmetric; it skips the test for pairs already joined.

## In practice

```rust,ignore
{{#include ../../../crates/sorting/examples/sorting_selection.rs}}
```

```text
top three scores: [100, 95, 95]
anagram classes: [["listen", "enlist", "inlets", "silent"], ["google", "elgoog"], ["banana"]]
chained clusters: [[1.0, 2.5, 4.0], [10.0, 11.5], [30.0]]
```

In the last line 1.0 and 4.0 are 3 apart, yet in the same class, linked through 2.5.

**Strengths**

- One-pass, low-memory top-$m$ selection for any ordered type.
- Equivalence classes from an arbitrary relation, with transitive closure and stable ordering.

**Weaknesses**

- No linear-time selection of a single order statistic (such as the median) and no smallest-$m$ variant; use `select_nth_unstable` from the standard library, or reverse the order.
- Equivalence classes are quadratic; there is no key-based fast path.

## Exercises

1. Show that the bounded heap finds the $m$ largest elements, and count its comparisons on a sorted input.
2. Estimate the expected number of heap replacements on a random permutation (the element at position $i$ enters with probability $m/i$).
3. Show that the transitive closure of a symmetric, reflexive relation is an equivalence relation.
4. Give a relation for which the result depends on the transitive closure, and explain why "within 2" clusters are unbounded in size.
5. Group $n$ words into anagram classes in $O(n \log n)$ by sorting with a canonical key.

## Further reading

- D. E. Knuth, *The Art of Computer Programming*, vol. 3, *Sorting and Searching*, 2nd ed., Addison-Wesley, 1998, sections 5.2.3 and 5.3.3.
- M. Blum, R. W. Floyd, V. Pratt, R. L. Rivest and R. E. Tarjan, "Time bounds for selection", *Journal of Computer and System Sciences* 7 (1973) 448–461.
- B. A. Galler and M. J. Fischer, "An improved equivalence algorithm", *Communications of the ACM* 7 (1964) 301–303.
- W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery, *Numerical Recipes*, 3rd ed., Cambridge University Press, 2007, section 8.5–8.6.
