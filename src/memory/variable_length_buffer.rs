use std::marker::PhantomData;
use std::sync::Arc;

use yarvk::binding_resource::BindingResource;
use yarvk::device::Device;
use yarvk::device_memory::{DeviceMemory, UnboundResource};
use yarvk::physical_device::memory_properties::MemoryType;
use yarvk::{BoundContinuousBuffer, Buffer, MemoryPropertyFlags};
use yarvk::{BufferUsageFlags, ContinuousBuffer, DeviceSize};

use crate::memory::auto_mapped_device_memory::AutoMappedDeviceMemory;
use crate::memory::private::PrivateMemoryBackedResource;
use crate::memory::{copy_buffer_from_start, MemoryBackedResource};

pub struct VariableLengthBuffer<T: Sized + 'static + Send + Sync> {
    len: usize,
    usage: BufferUsageFlags,
    buffer: BoundContinuousBuffer,
    device_memory: Arc<AutoMappedDeviceMemory>,
    end: usize,
    _phantom_data: PhantomData<T>,
}

impl<T: Sized + 'static + Send + Sync> VariableLengthBuffer<T> {
    pub fn new(
        device: &Arc<Device>,
        memory_type: &MemoryType,
        usage: BufferUsageFlags,
        len: usize,
    ) -> Self {
        assert!(memory_type
            .property_flags
            .contains(MemoryPropertyFlags::HOST_VISIBLE));
        let size = (len * std::mem::size_of::<T>()) as DeviceSize;
        let mut buffer_builder = ContinuousBuffer::builder(device);
        buffer_builder.usage(usage);
        buffer_builder.size(size);
        let buffer = buffer_builder.build().unwrap();
        let device_memory = DeviceMemory::builder(&memory_type, buffer.device())
            .allocation_size(size)
            .build()
            .unwrap();
        let buffer = buffer.bind_memory(&device_memory, 0).unwrap();
        let device_memory = Arc::new(AutoMappedDeviceMemory::new(device_memory));
        Self {
            len,
            usage,
            buffer,
            device_memory,
            end: 0,
            _phantom_data: Default::default(),
        }
    }
    pub fn clone_new(&self) -> Self {
        Self::new(
            self.device(),
            &self.device_memory.memory_type,
            self.usage,
            self.len,
        )
    }
    pub fn expand_to(&mut self, total_len: usize) {
        if total_len > self.len {
            let extra_size = total_len - self.len;
            self.expand(extra_size);
        }
    }
    pub fn expand(&mut self, len: usize) {
        let new_one = Self::new(
            &self.buffer.device,
            &self.device_memory.memory_type,
            self.usage,
            self.len + len,
        );
        copy_buffer_from_start(
            &self.buffer.device,
            &self.device_memory,
            &new_one.device_memory,
            self.buffer.size(),
        );
        self.device_memory = new_one.device_memory;
        self.buffer = new_one.buffer;
        self.len = new_one.len;
    }
    pub fn write(&mut self, data: &[T]) -> usize {
        let size = std::mem::size_of_val(data) as DeviceSize;
        let byte_offset = (self.end * std::mem::size_of::<T>()) as DeviceSize;
        if byte_offset >= self.device_memory.size || byte_offset + size > self.device_memory.size {
            panic!("buffer is full")
        }
        let data_in_u8 =
            unsafe { std::slice::from_raw_parts(data as *const _ as *const u8, size as _) };
        self.map_memory(
            byte_offset,
            size,
            Box::new(|slice: &mut [u8]| slice.copy_from_slice(data_in_u8)),
        )
        .unwrap();
        let offset = self.end;
        self.end += data.len();
        offset
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn clear(&mut self) {
        self.end = 0;
    }
}

impl<T: Sized + 'static + Send + Sync> BindingResource for VariableLengthBuffer<T> {
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

impl<T: Sized + 'static + Send + Sync> PrivateMemoryBackedResource for VariableLengthBuffer<T> {
    fn memory_property_flags(&self) -> MemoryPropertyFlags {
        self.device_memory.memory_type.property_flags
    }

    fn map_memory(
        &self,
        offset: DeviceSize,
        size: DeviceSize,
        f: Box<dyn FnOnce(&mut [u8])>,
    ) -> Result<(), yarvk::Result> {
        assert!(offset + size <= self.device_memory.size);
        let offset = self.offset() + offset;
        self.device_memory.map_memory(offset, size, f)
    }

    fn get_device_memory(&self) -> Arc<AutoMappedDeviceMemory> {
        self.device_memory.clone()
    }
}

impl<T: Sized + 'static + Send + Sync> MemoryBackedResource for VariableLengthBuffer<T> {}
