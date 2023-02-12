use std::collections::btree_map::RangeMut;
use std::collections::BTreeMap;
#[cfg(test)]
use std::ops::Deref;
use rustc_hash::FxHashSet;

#[derive(Default)]
pub struct UnusedBlocks(BTreeMap<u64 /*len*/, FxHashSet<u64>>);

impl UnusedBlocks {
    pub fn insert(
        &mut self,
        offset: u64,
        len: u64,
    ) {
        let blocks = &mut self.0;
        match blocks.get_mut(&len) {
            None => {
                let mut set = FxHashSet::default();
                set.insert(offset);
                blocks.insert(len, set);
            }
            Some(set) => {
                set.insert(offset);
            }
        }
    }
    pub fn remove(
        &mut self,
        offset: u64,
        len: u64,
    ) {
        let blocks = &mut self.0;
        if let Some(set) = blocks.get_mut(&len) {
            set.remove(&offset);
            if set.is_empty() {
                blocks.remove(&len);
            }
        };
    }
    pub fn get_fit_blocks(&mut self, len: u64) -> RangeMut<u64, FxHashSet<u64>> {
        self.0.range_mut(len..)
    }
}

#[cfg(test)]
impl Deref for UnusedBlocks {
    type Target = BTreeMap<u64 /*len*/, FxHashSet<u64>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}