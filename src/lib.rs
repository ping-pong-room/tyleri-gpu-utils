#![feature(trait_upcasting)]

use dashmap::{DashMap, DashSet};
use rustc_hash::FxHasher;
use std::hash::BuildHasherDefault;

pub mod memory;
pub mod queue;
pub mod thread;

type FxDashMap<K, V> = DashMap<K, V, BuildHasherDefault<FxHasher>>;
