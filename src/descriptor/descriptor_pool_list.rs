use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, Weak};
use yarvk::descriptor_set::desccriptor_pool::DescriptorPool;
use yarvk::descriptor_set::descriptor_set::{DescriptorSet, DescriptorSetValue};
use yarvk::descriptor_set::descriptor_set_layout::DescriptorSetLayout;

pub struct DescriptorPoolList<T: DescriptorSetValue> {
    pub descriptor_set_layout: Arc<DescriptorSetLayout<T>>,
    head: AtomicPtr<PoolNode<T>>,
}

struct PoolNode<T: DescriptorSetValue> {
    descriptor_pool: Weak<DescriptorPool<T>>,
    counts: u32,
    next: AtomicPtr<PoolNode<T>>,
}

impl<T: DescriptorSetValue> DescriptorPoolList<T> {
    pub fn new(descriptor_set_layout: &Arc<DescriptorSetLayout<T>>) -> Self {
        DescriptorPoolList {
            descriptor_set_layout: descriptor_set_layout.clone(),
            head: AtomicPtr::new(std::ptr::null_mut()),
        }
    }
    pub fn allocate(
        &self,
        mut counts: u32,
        descriptor_sets: &mut Vec<DescriptorSet<T>>,
    ) -> Result<(), yarvk::Result> {
        let mut ptr = self.head.load(Ordering::Relaxed);
        let mut pre_ptr: *mut PoolNode<T> = std::ptr::null_mut();
        unsafe {
            let biggest_counts = if ptr.is_null() { 0 } else { (*ptr).counts };
            while !ptr.is_null() {
                if let Some(pool) = (*ptr).descriptor_pool.upgrade() {
                    while counts != 0 {
                        if let Some(descriptor_set) = pool.allocate() {
                            counts -= 1;
                            descriptor_sets.push(descriptor_set);
                        } else {
                            break;
                        }
                    }
                    if counts == 0 {
                        break;
                    } else {
                        pre_ptr = ptr;
                        ptr = (*ptr).next.load(Ordering::Relaxed);
                    }
                } else if !pre_ptr.is_null() {
                    (*pre_ptr)
                        .next
                        .store((*ptr).next.load(Ordering::Relaxed), Ordering::Relaxed);
                    let unused_node = Box::from_raw(ptr);
                    ptr = unused_node.next.load(Ordering::Relaxed);
                } else {
                    break;
                }
            }
            if counts != 0 {
                let descriptor_pool = DescriptorPool::new(&self.descriptor_set_layout, counts)?;

                while counts != 0 {
                    let descriptor_set = descriptor_pool
                        .allocate()
                        .expect("WTF, descriptor pool should not fail here");
                    counts -= 1;
                    descriptor_sets.push(descriptor_set);
                }

                let new_node = Box::new(PoolNode {
                    descriptor_pool: Arc::downgrade(&descriptor_pool),
                    counts: std::cmp::max(counts, biggest_counts),
                    next: AtomicPtr::new(std::ptr::null_mut()),
                });
                let new_node_ptr = Box::into_raw(new_node);
                let mut head = self.head.load(Ordering::Relaxed);
                loop {
                    (*new_node_ptr).next.store(head, Ordering::Relaxed);
                    match self.head.compare_exchange_weak(
                        head,
                        new_node_ptr,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(h) => head = h,
                    }
                }
            }
            Ok(())
        }
    }
}

impl<T: DescriptorSetValue> Drop for DescriptorPoolList<T> {
    fn drop(&mut self) {
        let mut head = self.head.load(Ordering::Relaxed);
        while !head.is_null() {
            let node = unsafe { Box::from_raw(head) };
            head = node.next.load(Ordering::Relaxed);
        }
    }
}
