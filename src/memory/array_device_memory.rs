use crate::memory::auto_mapped_device_memory::AutoMappedDeviceMemory;
use crate::memory::private::PrivateMemoryBackedResource;
use crate::memory::{MemBakRes, MemoryBackedResource, MemoryObjectBuilder};
use crate::thread::parallel_vector_group::ParGroup;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::sync::Arc;
use yarvk::binding_resource::{BindMemoryInfo, BindingResource};
use yarvk::device::Device;
use yarvk::device_memory::{DeviceMemory, UnboundResource};
use yarvk::physical_device::memory_properties::MemoryType;
use yarvk::WHOLE_SIZE;
use yarvk::{DeviceSize, MemoryPropertyFlags};

struct ArrayMemoryObject<T: UnboundResource> {
    object: T::BoundType,
    device_memory: Arc<AutoMappedDeviceMemory>,
}

impl<T: UnboundResource> BindingResource for ArrayMemoryObject<T> {
    type RawTy = T::RawTy;

    fn raw(&self) -> &Self::RawTy {
        self.object.raw()
    }

    fn raw_mut(&mut self) -> &mut Self::RawTy {
        self.object.raw_mut()
    }

    fn offset(&self) -> DeviceSize {
        self.object.offset()
    }

    fn size(&self) -> DeviceSize {
        self.object.size()
    }

    fn device(&self) -> &Arc<Device> {
        self.object.device()
    }
}

impl<T: UnboundResource> PrivateMemoryBackedResource for ArrayMemoryObject<T> {
    fn memory_property_flags(&self) -> MemoryPropertyFlags {
        self.device_memory.memory_type.property_flags
    }
    fn memory_memory(
        &mut self,
        offset: DeviceSize,
        size: DeviceSize,
        f: &dyn Fn(&mut [u8]),
    ) -> Result<(), yarvk::Result> {
        assert!(offset + size <= self.size());
        let offset = self.offset() + offset;
        self.device_memory.map_memory(offset, size, f)?;
        Ok(())
    }

    fn get_device_memory(&self) -> Arc<AutoMappedDeviceMemory> {
        self.device_memory.clone()
    }
}

impl<T: UnboundResource> MemoryBackedResource for ArrayMemoryObject<T> {}

pub struct ArrayDeviceMemory {}
impl ArrayDeviceMemory {
    pub fn new_with_built<
        T: UnboundResource + 'static,
        Builder: Send + Sync + MemoryObjectBuilder<Ty = T>,
    >(
        device: &Arc<Device>,
        object_builder: &Builder,
        built: T,
        counts: usize,
        memory_type: &MemoryType,
    ) -> Result<Vec<Arc<MemBakRes<T::RawTy>>>, yarvk::Result> {
        let mut results = Vec::with_capacity(counts);
        let unbound = built;
        let memory_requirements = unbound.get_memory_requirements();
        let padding_size =
            memory_requirements.size + memory_requirements.size % memory_requirements.alignment;
        let mut device_memory = DeviceMemory::builder(memory_type, device)
            .allocation_size(padding_size * counts as u64)
            .build()?;
        let _map_result = device_memory.map_memory(0, WHOLE_SIZE);
        (1..counts)
            .into_par_iter()
            .map(|i| BindMemoryInfo {
                resource: object_builder.build().unwrap(),
                memory: &device_memory,
                memory_offset: i as DeviceSize * padding_size,
            })
            .collect_into_vec(&mut results);
        results.push(BindMemoryInfo {
            resource: unbound,
            memory: &device_memory,
            memory_offset: 0,
        });
        let mut results = results
            .split_for_par()
            .into_par_iter()
            .map(|chunk| T::bind_memories(device, chunk).unwrap())
            .collect::<Vec<_>>();
        let device_memory = Arc::new(AutoMappedDeviceMemory::new(device_memory));
        let mut vec = Vec::with_capacity(counts);
        while let Some(mut sub) = results.pop() {
            while let Some(item) = sub.pop() {
                vec.push(Arc::new(ArrayMemoryObject::<T> {
                    object: item,
                    device_memory: device_memory.clone(),
                }) as Arc<MemBakRes<T::RawTy>>);
            }
        }
        Ok(vec)
    }
    pub fn new_resources<
        T: UnboundResource + 'static,
        Builder: Send + Sync + MemoryObjectBuilder<Ty = T>,
    >(
        device: &Arc<Device>,
        object_builder: &Builder,
        counts: usize,
        memory_type: &MemoryType,
    ) -> Result<Vec<Arc<MemBakRes<T::RawTy>>>, yarvk::Result> {
        let unbound = object_builder.build()?;
        Self::new_with_built(device, object_builder, unbound, counts, memory_type)
    }
}
