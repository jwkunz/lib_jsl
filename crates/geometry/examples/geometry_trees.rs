//! Nearest-neighbour and range queries with a k-d tree, a quadtree and an octree.

use lib_jsl_geometry::algorithms::{KdTree, OctTree, QuadTree};

fn main() {
    let cities = [[0.0, 0.0], [3.0, 4.0], [-2.0, 1.0], [5.0, -1.0], [1.0, 1.0], [-4.0, -3.0], [2.5, 2.0]];
    let tree = KdTree::new(&cities);
    println!("nearest to (2, 2): {:?}", tree.nearest([2.0, 2.0]));
    println!("3 nearest to (0, 0): {:?}", tree.k_nearest([0.0, 0.0], 3));
    let mut near = tree.within_radius([0.0, 0.0], 2.5);
    near.sort_unstable();
    println!("within 2.5 of the origin: {near:?}");
    let mut boxed = tree.range_query([0.0, 0.0], [3.0, 4.0]);
    boxed.sort_unstable();
    println!("in the box [0, 3] × [0, 4]: {boxed:?}");

    // A quadtree takes insertions and removals; ids are chosen by the caller.
    let mut quad = QuadTree::new([-5.0, -5.0], [5.0, 5.0]);
    for (id, &p) in cities.iter().enumerate() {
        quad.insert(100 + id, p);
    }
    quad.remove(104, cities[4]);
    println!("quadtree nearest to (2, 2) after removing id 104: {:?}", quad.nearest([2.0, 2.0]));

    let cloud: Vec<[f64; 3]> = (0..1000).map(|i| { let t = i as f64 * 0.01; [t.cos(), t.sin(), 0.1 * t] }).collect();
    let oct = OctTree::from_points(&cloud);
    println!("octree of a helix: {} points, depth {}, nearest to (1, 0, 5): {:?}", oct.len(), oct.depth(), oct.nearest([1.0, 0.0, 5.0]));
}
