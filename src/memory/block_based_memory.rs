use crate::memory::auto_mapped_device_memory::AutoMappedDeviceMemory;
use crate::memory::block_based_memory::chunk::Chunk;
use crate::memory::private::PrivateMemoryBackedResource;
use crate::memory::{MemoryBackedResource, MemoryResource};
use derive_more::{Deref, DerefMut};
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};
use yarvk::binding_resource::BindingResource;
use yarvk::device::Device;
use yarvk::device_memory::{DeviceMemory, IMemoryRequirements, UnboundResource};
use yarvk::physical_device::memory_properties::MemoryType;
use yarvk::DeviceSize;
use yarvk::{MemoryPropertyFlags};

mod chunk;
mod unused_blocks;

struct BlockBasedResource<T: UnboundResource> {
    resource: T::BoundType,
    block_index: BlockIndex,
    allocator: Arc<BlockBasedAllocator>,
}

impl<T: UnboundResource> BindingResource for BlockBasedResource<T> {
    type RawTy = T::RawTy;

    fn raw(&self) -> &Self::RawTy {
        self.resource.raw()
    }

    fn raw_mut(&mut self) -> &mut Self::RawTy {
        self.resource.raw_mut()
    }

    fn offset(&self) -> DeviceSize {
        self.resource.offset()
    }

    fn size(&self) -> DeviceSize {
        self.resource.size()
    }

    fn device(&self) -> &Arc<Device> {
        todo!()
    }
}

impl<T: UnboundResource> PrivateMemoryBackedResource for BlockBasedResource<T> {
    fn memory_property_flags(&self) -> MemoryPropertyFlags {
        self.allocator.memory_type.property_flags
    }

    fn memory_memory(
        &mut self,
        offset: DeviceSize,
        size: DeviceSize,
        f: &dyn Fn(&mut [u8]),
    ) -> Result<(), yarvk::Result> {
        assert!(offset + size <= self.size());
        let offset = self.offset() + offset;
        let chunks = self.allocator.chunks.read().unwrap();
        chunks
            .get(&self.block_index.chunk_index)
            .unwrap()
            .device_memory
            .map_memory(offset, size, f)
    }

    fn get_device_memory(&self) -> Arc<AutoMappedDeviceMemory> {
        let chunks = self.allocator.chunks.read().unwrap();
        chunks
            .get(&self.block_index.chunk_index)
            .unwrap()
            .device_memory
            .clone()
    }
}

impl<T: UnboundResource> MemoryBackedResource for BlockBasedResource<T> {}

impl<T: UnboundResource> Drop for BlockBasedResource<T> {
    fn drop(&mut self) {
        self.allocator.free(&self.block_index)
    }
}

#[derive(PartialEq, Eq, Hash)]
struct BlockIndex {
    pub offset: u64,
    pub chunk_index: u64,
}

#[derive(Deref, DerefMut)]
struct VkChunk {
    #[deref]
    #[deref_mut]
    chunk: Chunk,
    device_memory: Arc<AutoMappedDeviceMemory>,
}

pub struct BlockBasedAllocator {
    device: Arc<Device>,
    memory_type: MemoryType,
    chunks: RwLock<BTreeMap<u64 /*size*/, VkChunk>>,
}

impl BlockBasedAllocator {
    pub fn new(device: &Arc<Device>, memory_type: MemoryType) -> Arc<Self> {
        Arc::new(Self {
            device: device.clone(),
            memory_type,
            chunks: Default::default(),
        })
    }
    pub fn capacity(&self, len: u64) -> Result<(), yarvk::Result> {
        // we allocate the space twice then asked, to make sure the memory is big enough to hold
        // resource with any alignment
        let len = len * 2;
        let device_memory = DeviceMemory::builder(&self.memory_type, &self.device)
            .allocation_size(len)
            .build()?;
        self.chunks.write().unwrap().insert(
            len,
            VkChunk {
                chunk: Chunk::new(len),
                device_memory: Arc::new(AutoMappedDeviceMemory::new(device_memory)),
            },
        );
        Ok(())
    }
    fn capacity_with_allocate(
        device: &Arc<Device>,
        memory_type: &MemoryType,
        chunks: &mut BTreeMap<u64 /*size*/, VkChunk>,
        len: u64,
        allocate_len: u64,
    ) -> Result<BlockIndex, yarvk::Result> {
        // we allocate the space twice then asked, to make sure the memory is big enough to hold
        // resource with any alignment
        let len = len * 2;
        let device_memory = DeviceMemory::builder(memory_type, device)
            .allocation_size(len)
            .build()?;
        let device_memory = Arc::new(AutoMappedDeviceMemory::new(device_memory));
        chunks.insert(
            len,
            VkChunk {
                chunk: Chunk::new_and_allocated(len, allocate_len),
                device_memory,
            },
        );
        let block_index = BlockIndex {
            offset: 0,
            chunk_index: len,
        };
        Ok(block_index)
    }
    pub fn allocate<T: UnboundResource + IMemoryRequirements + 'static>(
        self: &Arc<Self>,
        t: T,
    ) -> Result<Arc<MemoryResource<T::RawTy>>, yarvk::Result> {
        let memory_requirements = t.get_memory_requirements();
        let mut block_index = None;
        let block_index = {
            let mut chunks = self.chunks.write().unwrap();
            for (len, chunk) in chunks.iter_mut().rev() {
                match chunk.allocate(memory_requirements.size, memory_requirements.alignment) {
                    None => {
                        continue;
                    }
                    Some(offset) => {
                        block_index = Some(BlockIndex {
                            offset,
                            chunk_index: *len,
                        });
                        break;
                    }
                }
            }
            match block_index {
                None => {
                    let max_chunk_size = match chunks.iter().rev().next() {
                        None => memory_requirements.size,
                        Some((_, chunk)) => chunk.device_memory.size,
                    };

                    Self::capacity_with_allocate(
                        &self.device,
                        &self.memory_type,
                        &mut chunks,
                        max_chunk_size * 2,
                        memory_requirements.size,
                    )?
                }
                Some(block_index) => block_index,
            }
        };
        let chunks = self.chunks.read().unwrap();
        let chunk = chunks.get(&block_index.chunk_index).unwrap();
        let resource = t
            .bind_memory(&chunk.device_memory.get_device_memory(), block_index.offset)
            .expect("internal error: bind_memory failed");
        Ok(Arc::new(BlockBasedResource::<T> {
            resource,
            block_index,
            allocator: self.clone(),
        }))
    }
    fn free(&self, block_index: &BlockIndex) {
        let mut chunks = self.chunks.write().unwrap();
        if let Some(chunk) = chunks.get_mut(&block_index.chunk_index) {
            chunk.free(block_index.offset);
            if chunk.is_unused() {
                chunks.remove(&block_index.chunk_index);
            }
        }
    }
}
