#![feature(trait_upcasting)]

use dashmap::DashMap;
use rustc_hash::FxHasher;
use std::hash::BuildHasherDefault;

pub mod descriptor;
pub mod image;
pub mod memory;
pub mod queue;
pub mod thread;

type FxDashMap<K, V> = DashMap<K, V, BuildHasherDefault<FxHasher>>;
