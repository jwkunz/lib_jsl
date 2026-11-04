//! Electromagnetics: field containers on structured domains, antenna models, sensor arrays and
//! their signal models, beamformers and direction finding, transmission lines and two-port
//! networks, propagation and link budgets, spherical wave expansions and near-field to far-field
//! transformation, and two full-wave solvers (method of moments for wires, FDTD). See `DESIGN.md`
//! in this crate for the architecture and conventions.

pub mod algorithms;
pub mod antenna_models;
pub mod arrays;
pub mod conventions;
pub mod coordinates;
pub mod fields;
pub mod signal_models;
pub mod linalg;
pub mod networks;
pub mod propagation;
