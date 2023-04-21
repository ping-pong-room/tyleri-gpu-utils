use std::marker::PhantomData;
use std::slice::from_raw_parts_mut;
use std::sync::Arc;

use arc_swap::ArcSwap;
use parking_lot::Mutex;
use yarvk::binding_resource::BindingResource;
use yarvk::device::Device;
use yarvk::device_memory::{DeviceMemory, UnboundResource};
use yarvk::physical_device::memory_properties::MemoryType;
use yarvk::pipeline::pipeline_stage_flags::PipelineStageFlag;
use yarvk::{AccessFlags, DeviceSize};
use yarvk::{BoundContinuousBuffer, MemoryPropertyFlags};
use yarvk::{Buffer, BufferUsageFlags, ContinuousBuffer};

use crate::memory::auto_mapped_device_memory::AutoMappedDeviceMemory;
use crate::memory::block_based_memory::chunk::Chunk;
use crate::memory::memory_updater::MemoryUpdater;
use crate::memory::private::PrivateMemoryBackedResource;
use crate::memory::{
    cmd_copy_buffer_from_start, copy_buffer_from_start, IMemBakBuf, MemoryBackedResource,
};
use crate::queue::parallel_recording_queue::ParallelRecordingQueue;

struct DedicatedBuffer<T: Sized + 'static + Send + Sync> {
    buffer: BoundContinuousBuffer,
    device_memory: Arc<AutoMappedDeviceMemory>,
    _phantom_data: PhantomData<T>,
}

impl<T: Sized + 'static + Send + Sync> DedicatedBuffer<T> {
    fn new(
        device: &Arc<Device>,
        memory_type: &MemoryType,
        usage: BufferUsageFlags,
        len: usize,
    ) -> Result<Self, yarvk::Result> {
        let size = (len * std::mem::size_of::<T>()) as DeviceSize;
        let mut buffer_builder = ContinuousBuffer::builder(device);
        buffer_builder.size(size);
        buffer_builder.usage(usage);
        let buffer = buffer_builder.build()?;
        let device_memory = DeviceMemory::builder(&memory_type, buffer.device())
            .allocation_size(size)
            .build()?;
        let buffer = buffer.bind_memory(&device_memory, 0)?;
        let device_memory = Arc::new(AutoMappedDeviceMemory::new(device_memory));
        Ok(Self {
            buffer,
            device_memory,
            _phantom_data: Default::default(),
        })
    }
}

impl<T: Sized + 'static + Send + Sync> BindingResource for DedicatedBuffer<T> {
    type RawTy = Buffer;

    fn raw(&self) -> &Self::RawTy {
        self.buffer.raw()
    }

    fn raw_mut(&mut self) -> &mut Self::RawTy {
        self.buffer.raw_mut()
    }

    fn offset(&self) -> DeviceSize {
        0
    }

    fn size(&self) -> DeviceSize {
        self.device_memory.size
    }

    fn device(&self) -> &Arc<Device> {
        self.buffer.device()
    }
}

impl<T: Sized + 'static + Send + Sync> PrivateMemoryBackedResource for DedicatedBuffer<T> {
    fn memory_property_flags(&self) -> MemoryPropertyFlags {
        self.device_memory.memory_type.property_flags
    }

    fn map_memory(
        &self,
        offset: DeviceSize,
        size: DeviceSize,
        f: Box<dyn FnOnce(&mut [u8])>,
    ) -> Result<(), yarvk::Result> {
        assert!(offset + size <= self.size());
        let offset = self.offset() + offset;
        self.device_memory.map_memory(offset, size, f)
    }

    fn get_device_memory(&self) -> Arc<AutoMappedDeviceMemory> {
        self.device_memory.clone()
    }
}

impl<T: Sized + 'static + Send + Sync> MemoryBackedResource for DedicatedBuffer<T> {}

#[derive(Clone)]
pub struct BindlessBuffer<T: Sized + 'static + Send + Sync> {
    pub offset: usize,
    pub len: usize,
    pub(crate) bindless_buffer: Arc<BindlessBufferAllocator<T>>,
}

impl<T: Sized + 'static + Send + Sync> Drop for BindlessBuffer<T> {
    fn drop(&mut self) {
        self.bindless_buffer
            .chunk
            .lock()
            .free((self.offset * std::mem::size_of::<T>()) as _);
    }
}

pub struct BindlessBufferAllocator<T: Sized + 'static + Send + Sync> {
    usage: BufferUsageFlags,
    dedicated_buffer: ArcSwap<DedicatedBuffer<T>>,
    chunk: Mutex<Chunk>,
}

impl<T: Sized + 'static + Send + Sync> BindlessBufferAllocator<T> {
    pub fn new(
        device: &Arc<Device>,
        len: usize,
        memory_type: &MemoryType,
        usage: BufferUsageFlags,
    ) -> Result<Arc<Self>, yarvk::Result> {
        let dedicated_buffer = ArcSwap::from(Arc::new(DedicatedBuffer::new(
            device,
            memory_type,
            usage,
            len,
        )?));
        let chunk = Mutex::new(Chunk::new((len * std::mem::size_of::<T>()) as _));
        Ok(Arc::new(Self {
            usage,
            dedicated_buffer,
            chunk,
        }))
    }
    pub fn get_buffer(&self) -> Arc<IMemBakBuf> {
        self.dedicated_buffer.load().clone()
    }
    pub fn allocate<'a>(
        self: &Arc<Self>,
        data: Vec<(usize /*len*/, Box<dyn FnOnce(&mut [T]) + Send + Sync>)>,
        queue: &mut ParallelRecordingQueue,
    ) -> Vec<Arc<BindlessBuffer<T>>> {
        let mut chunk = self.chunk.lock();
        let mut result_buffers = vec![
            Arc::new(BindlessBuffer {
                offset: 0,
                len: 0,
                bindless_buffer: self.clone(),
            });
            data.len()
        ];
        let mut unallocated = Vec::new();
        let mut not_allocated_len = 0;
        let dedicated_buffer = self.dedicated_buffer.load();
        let buffer = &dedicated_buffer.buffer;
        data.iter().enumerate().for_each(|(index, (len, _))| {
            match chunk.allocate((len * std::mem::size_of::<T>()) as _, 1 as _) {
                Some(offset) => {
                    let buffer = Arc::get_mut(&mut result_buffers[index]).unwrap();
                    buffer.offset = offset as usize / std::mem::size_of::<T>();
                    buffer.len = *len;
                }
                None => {
                    unallocated.push(index);
                    not_allocated_len += len;
                }
            }
        });
        if !unallocated.is_empty() {
            let device_memory = &dedicated_buffer.device_memory;
            let new_alloc_len = std::cmp::max(
                buffer.size() as usize / std::mem::size_of::<T>(),
                not_allocated_len,
            );
            let new_dedicated_buffer = DedicatedBuffer::new(
                &buffer.device,
                &device_memory.memory_type,
                self.usage,
                new_alloc_len,
            )
            .unwrap();
            let new_dedicated_buffer = Arc::new(new_dedicated_buffer);
            chunk.expand((new_alloc_len * std::mem::size_of::<T>()) as _);
            for index in unallocated {
                let len = data[index].0;
                let offset = chunk
                    .allocate((len * std::mem::size_of::<T>()) as _, 1 as _)
                    .unwrap();
                let buffer = Arc::get_mut(&mut result_buffers[index]).unwrap();
                buffer.offset = offset as usize / std::mem::size_of::<T>();
                buffer.len = len;
            }
            self.dedicated_buffer.swap(new_dedicated_buffer.clone());
            // can be parallel with above
            if dedicated_buffer
                .device_memory
                .memory_type
                .property_flags
                .contains(MemoryPropertyFlags::HOST_VISIBLE)
            {
                copy_buffer_from_start(
                    &buffer.device,
                    &dedicated_buffer.device_memory,
                    &new_dedicated_buffer.device_memory,
                    dedicated_buffer.buffer.size(),
                )
            } else {
                cmd_copy_buffer_from_start(
                    dedicated_buffer.clone(),
                    new_dedicated_buffer.clone(),
                    dedicated_buffer.size(),
                    queue,
                );
                queue.simple_submit().unwrap();
            }
        }
        let updater = MemoryUpdater::default();
        result_buffers
            .iter_mut()
            .zip(data)
            .into_iter()
            .for_each(|(bindless_buffer, (len, f))| {
                updater.add_bindless_buffer(
                    bindless_buffer,
                    0,
                    (len * std::mem::size_of::<T>()) as _,
                    AccessFlags::TRANSFER_WRITE,
                    PipelineStageFlag::Transfer.into(),
                    Box::new(move |slice: &mut [u8]| {
                        let slice = unsafe {
                            from_raw_parts_mut(
                                slice as *mut _ as *mut T,
                                slice.len() / std::mem::size_of::<T>(),
                            )
                        };
                        f(slice)
                    }),
                )
            });
        updater.update(queue);
        result_buffers
    }
}
