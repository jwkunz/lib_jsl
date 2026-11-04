# Sorting and selection

Sorting puts data in order so that it can be searched, merged, ranked and grouped; selection finds part of that order without paying for all of it. This chapter covers `lib_jsl_sorting`:
- three in-place comparison sorts: quicksort, heapsort and Shell's method;
- selecting the $m$ largest elements with a bounded heap;
- grouping items into equivalence classes.

## The central questions

- **Average or worst case?** Quicksort is fastest on average; heapsort guarantees $O(n \log n)$; Shell's method is simplest and adapts to nearly sorted data.
- **All of the order, or part of it?** For the top $m$ of $n$, a heap costs $O(n \log m)$ instead of $O(n \log n)$.
- **The library or the standard library?** Rust's own `sort` and `sort_unstable` are faster than any of these on primitive keys, and stable when needed. The algorithms here are clear implementations of the classics, to read, compare and teach from.

## Testing

Each sort is run on a million keys in five arrangements (random, nearly sorted, sorted, reversed, and only three distinct values), and its output is checked against the standard library's. Comparisons are counted by sorting a key type whose `Ord` implementation counts its calls.

Writing this chapter fixed quicksort, which took quadratic time on runs of equal keys. Its documentation also called its partition Hoare-style when it was Lomuto's. It now uses Hoare's partition.

## In the library

All of these are in `lib_jsl_sorting` ([API reference](../../api/lib_jsl_sorting/index.html)).

| You need | Use | Entry |
|---|---|---|
| A slice in order | `quicksort`, `heapsort`, `shell_sort` | [Comparison sorting](comparison-sorts.md) |
| The largest few elements | `select_m_largest` | [Selection and equivalence classes](selection.md#selecting-the-m-largest) |
| Groups of equivalent items | `equivalence_classes` | [Selection and equivalence classes](selection.md#equivalence-classes) |
