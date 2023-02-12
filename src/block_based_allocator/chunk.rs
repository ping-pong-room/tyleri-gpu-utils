use rustc_hash::FxHashMap;
use crate::block_based_allocator::unused_blocks::UnusedBlocks;

struct Block {
    len: u64,
    used: bool,
    pre: Option<u64>,
    next: Option<u64>,
}

pub struct Chunk {
    unused_blocks: UnusedBlocks,
    blocks: FxHashMap<u64 /*offset*/, Block>,
}

impl Chunk {
    pub fn new(len: u64) -> Self {
        assert_ne!(len, 0);
        let mut blocks = FxHashMap::default();
        blocks.insert(
            0,
            Block {
                len,
                used: false,
                pre: None,
                next: None,
            },
        );
        let mut unused_blocks = UnusedBlocks::default();
        unused_blocks.insert(0, len);
        Self {
            unused_blocks,
            blocks,
        }
    }

    pub fn new_and_allocated(len: u64, allocated_len: u64) -> Self {
        assert_ne!(len, 0);
        assert_ne!(allocated_len, 0);
        let perfect_match = len == allocated_len;
        let mut blocks = FxHashMap::default();
        blocks.insert(
            0,
            Block {
                len: allocated_len,
                used: true,
                pre: None,
                next: if perfect_match {
                    None
                } else {
                    Some(allocated_len)
                },
            },
        );
        if !perfect_match {
            blocks.insert(
                allocated_len,
                Block {
                    len: len - allocated_len,
                    used: false,
                    pre: Some(0),
                    next: None,
                },
            );
        }
        let mut unused_blocks = UnusedBlocks::default();
        unused_blocks.insert(
            allocated_len,
            len - allocated_len,
        );
        Self {
            unused_blocks,
            blocks,
        }
    }
    pub fn allocate(&mut self, len: u64, alignment: u64) -> Option<u64/*offset*/> {
        if len == 0 {
            return None;
        }
        let result = None;
        // find available unused blocks
        for (unused_len, set) in self.unused_blocks.get_fit_blocks(len) {
            let unused_len = *unused_len;
            for offset in set.iter() {
                let offset = *offset;
                let block = self.blocks.get_mut(&offset)?;
                let offset_mod_alignment = offset % alignment;
                if offset_mod_alignment == 0 {
                    if block.len < len {
                        // too small
                        continue;
                    } else {
                        self.unused_blocks.remove(
                            offset,
                            unused_len,
                        );
                        block.used = true;
                        if len != block.len {
                            let new_offset = offset + len;
                            let new_len = block.len - len;
                            let new_block = Block {
                                len: new_len,
                                used: false,
                                pre: Some(offset),
                                next: block.next,
                            };
                            block.next = Some(new_offset);
                            block.len = len;
                            self.blocks.insert(new_offset, new_block);
                            self.unused_blocks.insert(
                                new_offset,
                                new_len,
                            );
                        }
                        return Some(offset);
                    }
                } else {
                    let wasted_len = alignment - offset_mod_alignment;
                    let available_len = block.len - wasted_len;
                    if available_len < len {
                        // not enough for alignment
                        continue;
                    } else {
                        self.unused_blocks.remove(
                            offset,
                            unused_len,
                        );
                        let block_len = block.len;
                        block.len = wasted_len;

                        let new_offset = offset + wasted_len;
                        let new_pre = Some(offset);
                        let new_next = block.next;
                        block.next = Some(new_offset);
                        let mut new_block = Block {
                            len,
                            used: true,
                            pre: new_pre,
                            next: new_next,
                        };
                        self.unused_blocks.insert(
                            offset,
                            block.len,
                        );

                        if available_len > len {
                            let tail_block_offset = offset + wasted_len + len;
                            let tail_block_len = block_len - wasted_len - len;
                            self.blocks.insert(
                                tail_block_offset,
                                Block {
                                    len: tail_block_len,
                                    used: false,
                                    pre: Some(new_offset),
                                    next: new_next,
                                },
                            );
                            new_block.next = Some(tail_block_offset);
                            self.unused_blocks.insert(
                                tail_block_offset,
                                tail_block_len,
                            );
                        }

                        self.blocks.insert(new_offset, new_block);
                        return Some(new_offset);
                    }
                }
            }
            continue;
        }
        result
    }
    pub fn free(&mut self, offset: u64) {
        unsafe {
            if let Some(mut block) = self.blocks.remove(&offset) {
                self.merge_with_next(&mut block);
                self.merge_with_pre(offset, block);
            }
        }
    }
    unsafe fn merge_with_pre(&mut self, offset: u64, mut block: Block) {
        if let Some(pre_offset) = block.pre {
            let pre_block = self.blocks.get_mut(&pre_offset).unwrap_unchecked();
            // if pre is unused.
            if !pre_block.used {
                self.unused_blocks.remove(
                    pre_offset,
                    pre_block.len,
                );
                pre_block.len += block.len;
                pre_block.next = block.next;
                self.unused_blocks.insert(
                    pre_offset,
                    pre_block.len,
                );
                return;
            }
        }
        self.unused_blocks.insert(offset, block.len);
        block.used = false;
        self.blocks.insert(offset, block);
    }
    unsafe fn merge_with_next(&mut self, block: &mut Block) {
        if let Some(i) = block.next {
            let next_block = self.blocks.get(&i).unwrap_unchecked();
            if !next_block.used {
                block.len += next_block.len;
                block.next = next_block.next;
                self.unused_blocks.remove(i, next_block.len);
                self.blocks.remove(&i);
            }
        }
    }
    pub fn is_unused(&self) -> bool {
        if self.blocks.len() == 1 {
            let block = unsafe { self.blocks.iter().next().unwrap_unchecked() };
            if !block.1.used {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
fn iter_blocks(blocks: &FxHashMap<u64 /*offset*/, Block>) -> Vec<u64> {
    let mut p_next = blocks.get(&0).unwrap().next;
    let block = blocks.get(&0).unwrap();
    let mut vec = Vec::new();
    let mut last_used = block.used;
    vec.push(0);
    while p_next.is_some() {
        let next_offset = p_next.unwrap();
        let next_block = blocks.get(&next_offset).unwrap();
        p_next = next_block.next;
        if !last_used && !next_block.used {
            panic!("two connected unused block")
        }
        last_used = next_block.used;
        vec.push(next_offset)
    }
    vec
}

#[test]
fn test() {
    let mut chunk = Chunk::new_and_allocated(128, 1);
    assert_eq!(chunk.blocks.get(&0).unwrap().len, 1);
    assert_eq!(chunk.blocks.get(&1).unwrap().len, 127);
    assert!(chunk
        .unused_blocks
        .get(&127)
        .unwrap()
        .contains(&1));
    assert_eq!(chunk.blocks.get(&0).unwrap().pre, None);
    assert_eq!(chunk.blocks.get(&0).unwrap().next, Some(1));
    assert_eq!(chunk.blocks.get(&1).unwrap().pre, Some(0));
    assert_eq!(chunk.blocks.get(&1).unwrap().next, None);

    chunk.allocate(2, 2);
    assert_eq!(chunk.blocks.get(&0).unwrap().len, 1);
    assert_eq!(chunk.blocks.get(&1).unwrap().len, 1);
    assert_eq!(chunk.blocks.get(&2).unwrap().len, 2);
    assert_eq!(chunk.blocks.get(&4).unwrap().len, 124);
    assert!(chunk
        .unused_blocks
        .get(&1)
        .unwrap()
        .contains(&1));
    assert!(chunk
        .unused_blocks
        .get(&124)
        .unwrap()
        .contains(&4));
    assert_eq!(chunk.blocks.get(&1).unwrap().pre, Some(0));
    assert_eq!(chunk.blocks.get(&1).unwrap().next, Some(2));
    assert_eq!(chunk.blocks.get(&2).unwrap().pre, Some(1));
    assert_eq!(chunk.blocks.get(&2).unwrap().next, Some(4));
    assert_eq!(chunk.blocks.get(&4).unwrap().pre, Some(2));
    assert_eq!(chunk.blocks.get(&4).unwrap().next, None);

    chunk.allocate(32, 32);
    assert_eq!(iter_blocks(&chunk.blocks).as_slice(), [0, 1, 2, 4, 32, 64]);

    chunk.allocate(8, 8);
    assert_eq!(
        iter_blocks(&chunk.blocks).as_slice(),
        [0, 1, 2, 4, 8, 16, 32, 64]
    );
    assert_eq!(chunk.unused_blocks.len(), 4);

    chunk.free(8);
    chunk.free(32);

    chunk.free(2);
    assert_eq!(chunk.blocks.get(&0).unwrap().len, 1);
    assert_eq!(chunk.blocks.get(&1).unwrap().len, 127);
    assert_eq!(chunk.blocks.len(), 2);
    assert!(chunk
        .unused_blocks
        .get(&127)
        .unwrap()
        .contains(&1));
    assert_eq!(chunk.unused_blocks.len(), 1);

    chunk.free(0);

    assert_eq!(chunk.blocks.get(&0).unwrap().len, 128);
    assert!(chunk
        .unused_blocks
        .get(&128)
        .unwrap()
        .contains(&0));
    assert_eq!(chunk.blocks.get(&0).unwrap().pre, None);
    assert_eq!(chunk.blocks.get(&0).unwrap().next, None);
}

#[test]
fn multiple_candidate_test() {
    let mut chunk = Chunk::new(128);
    for _i in 0..4 {
        chunk.allocate(1, 1);
    }
    assert_eq!(iter_blocks(&chunk.blocks).as_slice(), [0, 1, 2, 3, 4]);
    chunk.allocate(2, 2);
    chunk.allocate(1, 1);
    assert_eq!(iter_blocks(&chunk.blocks).as_slice(), [0, 1, 2, 3, 4, 6, 7]);
    chunk.free(1);
    assert_eq!(iter_blocks(&chunk.blocks).as_slice(), [0, 1, 2, 3, 4, 6, 7]);
    chunk.free(2);
    assert_eq!(iter_blocks(&chunk.blocks).as_slice(), [0, 1, 3, 4, 6, 7]);
    chunk.free(4);

    assert_eq!(iter_blocks(&chunk.blocks).as_slice(), [0, 1, 3, 4, 6, 7]);
    assert_eq!(chunk.unused_blocks.get(&2).unwrap().len(), 2);
    assert_eq!(chunk.allocate(2, 2), Some(4))
}