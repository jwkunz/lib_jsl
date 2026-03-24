//! Concrete keyed geometry table implementations used by the public API.
//!
//! These types are the storage backbone for the concrete geometry graph. They provide a
//! hash-backed implementation of [`IsGeometryTable`](crate::geometry::common::IsGeometryTable)
//! along with a shared-handle wrapper for cases where multiple primitives need to borrow the same
//! table.

use crate::geometry::common::{GeometricPrimitive, IsGeometryKey, IsGeometryTable};
use serde::Serialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

/// Shared interior-mutable handle to a hash-backed geometry table.
///
/// This alias is used throughout the concrete API so multiple primitives can reference the same
/// logical table without taking ownership of it.
pub type SharedGeometryTable<K, V> = Rc<RefCell<HashGeometryTable<K, V>>>;

/// Hash-backed keyed geometry table implementation.
///
/// This is the default concrete table type used by [`GeometryTableRegistry`](crate::geometry::registry::GeometryTableRegistry).
#[derive(Debug, Clone, Serialize)]
pub struct HashGeometryTable<K, V> {
    entries: HashMap<K, V>,
}

impl<K, V> Default for HashGeometryTable<K, V> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }
}

impl<K, V> HashGeometryTable<K, V> {
    /// Creates an empty hash-backed geometry table.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<K, V> IsGeometryTable for HashGeometryTable<K, V>
where
    K: IsGeometryKey,
    V: GeometricPrimitive,
{
    type Key = K;
    type Item = V;

    fn get(&self, key: &Self::Key) -> Option<Self::Item> {
        self.entries.get(key).cloned()
    }

    fn insert(&mut self, key: Self::Key, value: Self::Item) -> Result<(), String> {
        self.entries.insert(key, value);
        Ok(())
    }

    fn remove(&mut self, key: &Self::Key) -> Option<Self::Item> {
        self.entries.remove(key)
    }

    fn contains_key(&self, key: &Self::Key) -> bool {
        self.entries.contains_key(key)
    }

    fn size(&self) -> usize {
        self.entries.len()
    }

    fn union(&self, other: &Self) -> Self {
        let mut entries = self.entries.clone();
        entries.extend(other.entries.clone());
        Self { entries }
    }

    fn intersection(&self, other: &Self) -> Self {
        let entries = self
            .entries
            .iter()
            .filter(|(key, _)| other.entries.contains_key(*key))
            .map(|(key, value)| (*key, value.clone()))
            .collect();
        Self { entries }
    }

    fn iter(&self) -> Box<dyn Iterator<Item = (Self::Key, Self::Item)> + '_> {
        Box::new(self.entries.iter().map(|(key, value)| (*key, value.clone())))
    }

    fn to_file(&self, path: &str) -> Result<(), String> {
        let serialized = serde_json::to_string_pretty(self).map_err(|err| err.to_string())?;
        std::fs::write(Path::new(path), serialized).map_err(|err| err.to_string())
    }

    fn from_file(path: &str) -> Result<Self, String>
    where
        Self: Sized,
    {
        let _ = path;
        Err(
            "from_file is not implemented for generic HashGeometryTable without a concrete deserializer"
                .to_string(),
        )
    }
}

impl<K, V> IsGeometryTable for SharedGeometryTable<K, V>
where
    K: IsGeometryKey,
    V: GeometricPrimitive,
{
    type Key = K;
    type Item = V;

    fn get(&self, key: &Self::Key) -> Option<Self::Item> {
        self.borrow().get(key)
    }

    fn insert(&mut self, key: Self::Key, value: Self::Item) -> Result<(), String> {
        self.borrow_mut().insert(key, value)
    }

    fn remove(&mut self, key: &Self::Key) -> Option<Self::Item> {
        self.borrow_mut().remove(key)
    }

    fn contains_key(&self, key: &Self::Key) -> bool {
        self.borrow().contains_key(key)
    }

    fn size(&self) -> usize {
        self.borrow().size()
    }

    fn union(&self, other: &Self) -> Self {
        Rc::new(RefCell::new(self.borrow().union(&other.borrow())))
    }

    fn intersection(&self, other: &Self) -> Self {
        Rc::new(RefCell::new(self.borrow().intersection(&other.borrow())))
    }

    fn iter(&self) -> Box<dyn Iterator<Item = (Self::Key, Self::Item)> + '_> {
        let items: Vec<_> = self.borrow().iter().collect();
        Box::new(items.into_iter())
    }

    fn to_file(&self, path: &str) -> Result<(), String> {
        self.borrow().to_file(path)
    }

    fn from_file(path: &str) -> Result<Self, String>
    where
        Self: Sized,
    {
        HashGeometryTable::from_file(path).map(|table| Rc::new(RefCell::new(table)))
    }
}
